"""Image -> svg conversion."""
from colorsys import rgb_to_hls
import math
import re
import random
import tempfile
import xml.etree.ElementTree as etree

import cv2
from kmeans_gpu import KMeans
import numpy as np
import pydiffvg
import torch
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

import utils


def quantize_colors(image, n_clusters):
  restarts = 3

  if image.ndim == 3:
    image = image.unsqueeze(0)

  pixels = image.permute(0, 2, 3, 1) * 255
  pixels = pixels.reshape(image.shape[0], -1, 3).to(torch.float32)
  pixels = torch.cat([pixels] * restarts, 0)

  kmeans = KMeans(
      n_clusters=n_clusters,
      max_iter=50,
      tolerance=0.0001,
      distance='euclidean',
      sub_sampling=None,
  )

  with torch.no_grad():
    centroids = kmeans(pixels)

  centroids = centroids.to(torch.uint8).to(torch.float32)
  expanded_pixels = pixels.unsqueeze(2)  # [restarts, num_pixels, 1, 3]
  expanded_centroids = centroids.unsqueeze(1)  # [restarts, 1, num_clusters, 3]

  distances = torch.mean(torch.square(expanded_pixels - expanded_centroids), dim=-1)  # [restarts, num_pixels, num_clusters]
  distances, assignments = distances.min(-1)
  best_restart = distances.mean(-1).min(-1).indices

  quantized_pixels = torch.gather(centroids.to(torch.uint8)[best_restart][None], dim=1,
                                  index=assignments[best_restart][None].unsqueeze(-1).expand(-1, -1, 3))
  width = height = int(pixels.shape[1]**0.5)
  reshaped_pixels = quantized_pixels.view(-1, width, height, 3).permute(0, 3, 1, 2)

  return reshaped_pixels


def to_hex(color):
  return f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'


def random_polygon(canvas_size, num_sides):
  cx = random.randint(0, canvas_size)
  cy = random.randint(0, canvas_size)
  radius = random.randint(int(0.2 * canvas_size), int(0.4 * canvas_size))

  points = []
  for i in range(num_sides):
      angle = 2 * 3.14159 * i / num_sides
      x = int(cx + radius * random.uniform(0.8, 1.2) * math.cos(angle))
      y = int(cy + radius * random.uniform(0.8, 1.2) * math.sin(angle))
      x = max(0, min(canvas_size, x))
      y = max(0, min(canvas_size, y))
      points.append(f"{x},{y}")
  color = '#D3D3D3'
  opacity = 0.1
  return (f'<polygon points="{" ".join(points)}" fill="{color}" opacity="{opacity}"/>',
          f'<path d="M{" ".join(points)}Z" fill="{color}" opacity="{opacity}"/>')


def make_rough_svg(image, max_bytes):
  num_colors = 12
  for _ in range(3):
    svg, sizing_svg = make_heuristic_svg(image, max_bytes, num_colors)
    num_colors = int(num_colors * 1.5)
    if len(sizing_svg) > 0.8 * max_bytes:
      break
    #print("SVG not long enough, trying with more colors")
  footer = "</svg>"
  svg = svg.replace(footer, "")
  sizing_svg = sizing_svg.replace(footer, "")
  while True:
    polygon_str, path_str = random_polygon(384, 8)
    if len((sizing_svg + path_str + footer).encode('utf-8')) > max_bytes:
      break
    #print("Added random shape to increase SVG length")
    svg += polygon_str
    sizing_svg += path_str
  svg += footer
  sizing_svg += footer
  return svg, sizing_svg


def make_heuristic_svg(image, max_bytes, num_colors=12):
  header = '<svg xmlns="http://www.w3.org/2000/svg" width="384" height="384" viewBox="0 0 384 384">'
  footer = '</svg>'

  quantized = quantize_colors(utils.image_to_tensor(image), num_colors)[0]
  quantized = quantized.permute((1, 2, 0))
  pixels = quantized.squeeze(0).reshape(-1, 3).contiguous()
  colors, counts = torch.unique(pixels, return_counts=True, dim=0)
  most_common = to_hex(colors[counts.argmax()])
  background = f'<rect width="{384}" height="{384}" fill="{most_common}"/>'

  center_x, center_y = 384/2, 384/2

  polygons = []
  quantized = quantized.cpu().numpy()
  colors = [c.cpu().numpy() for c in colors]
  for color in colors:
    color_hex = to_hex(color)

    color_mask = cv2.inRange(quantized, color, color)
    contours, _ = cv2.findContours(
        color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
      area = cv2.contourArea(contour)
      if area < 10:
        continue

      m = cv2.moments(contour)
      if m["m00"] == 0:
        continue
      cx = int(m["m10"] / m["m00"])
      cy = int(m["m01"] / m["m00"])
      dist_from_center = (np.sqrt(((cx - center_x) / 384)**2 +
                          ((cy - center_y) / 384)**2))
      approx = []
      epsilon = 0.018 * (20 + cv2.arcLength(contour, True))
      for _ in range(10):
        approx = cv2.approxPolyDP(contour, epsilon, True)
        epsilon *= 0.75
        if len(approx) > 3:
          break
      if len(approx) < 3:
        continue

      importance = (
          area *
          (1.5 - dist_from_center) *
          (1 / (len(approx) + 1)) *
          (0.5 if to_hex(color) == most_common else 1)
      )
      polygons.append({
          "importance": importance,
          "points": approx,
          "color": color_hex,
          "color_raw": color,
          "area": area,
          "contour": contour,
      })

  svg = header + background
  sizing_svg = header + background

  for i, polygon in enumerate(sorted(polygons, key=lambda x: x["importance"], reverse=True)):
    polygon_str = '<polygon points="'
    path_str = '<path d="M'

    polygon_str += " ".join([f"{p[0][0]} {p[0][1]}" for p in polygon["points"]])
    path_str += " ".join(["999 999" for p in polygon["points"]])

    path_str += f'Z" fill="{polygon["color"]}" opacity=".00"/>'
    polygon_str += f'" fill="{polygon["color"]}" opacity="1"/>'

    if len((sizing_svg + path_str + footer).encode('utf-8')) > max_bytes:
      continue

    svg += polygon_str
    sizing_svg += path_str

  svg += footer
  sizing_svg += footer
  return svg, sizing_svg


def add_ocr_decoy(svg_code: str, corner=None) -> str:
    """OCR devcoy trick from https://www.kaggle.com/code/richolson/let-s-defeat-ocr-easy-lb-boost."""
    # Check if SVG has a closing tag
    if "</svg>" not in svg_code:
        return svg_code

    # Extract viewBox if it exists to understand the dimensions
    viewbox_match = re.search(r'viewBox=["\'](.*?)["\']', svg_code)
    if viewbox_match:
        viewbox = viewbox_match.group(1).split()
        try:
            x, y, width, height = map(float, viewbox)
        except ValueError:
            # Default dimensions if we can't parse viewBox
            width, height = 384, 384
    else:
        # Default dimensions if viewBox not found
        width, height = 384, 384

    # Function to convert hex color to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])
        return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))

    # Function to convert RGB to hex
    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )

    # Function to calculate color lightness
    def get_lightness(color):
        # Handle different color formats
        if color.startswith('#'):
            rgb = hex_to_rgb(color)
            return rgb_to_hls(*rgb)[1]  # Lightness is the second value in HLS
        elif color.startswith('rgb'):
            rgb_match = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color)
            if rgb_match:
                r, g, b = map(lambda x: int(x)/255, rgb_match.groups())
                return rgb_to_hls(r, g, b)[1]
        return 0.5  # Default lightness if we can't parse

    # Extract all colors from the SVG
    color_matches = re.findall(r'(?:fill|stroke)="(#[0-9A-Fa-f]{3,6}|rgb\(\d+,\s*\d+,\s*\d+\))"', svg_code)

    # Default colors in case we don't find enough
    second_darkest_color = "#333333"  # Default to dark gray
    second_brightest_color = "#CCCCCC"  # Default to light gray

    if color_matches:
        # Remove duplicates and get unique colors
        unique_colors = list(set(color_matches))

        # Calculate lightness for each unique color
        colors_with_lightness = [(color, get_lightness(color)) for color in unique_colors]

        # Sort by lightness (brightness)
        sorted_colors = sorted(colors_with_lightness, key=lambda x: x[1])

        # Handle different scenarios based on number of unique colors
        if len(sorted_colors) >= 4:
            # We have at least 4 unique colors - use 2nd darkest and 2nd brightest
            second_darkest_color = sorted_colors[1][0]
            second_brightest_color = sorted_colors[-2][0]
        elif len(sorted_colors) == 3:
            # We have 3 unique colors - use 2nd darkest and brightest
            second_darkest_color = sorted_colors[1][0]
            second_brightest_color = sorted_colors[2][0]
        elif len(sorted_colors) == 2:
            # We have only 2 unique colors - use the darkest and brightest
            second_darkest_color = sorted_colors[0][0]
            second_brightest_color = sorted_colors[1][0]
        elif len(sorted_colors) == 1:
            # Only one color - use it for second_darkest and a derived lighter version
            base_color = sorted_colors[0][0]
            base_lightness = sorted_colors[0][1]
            second_darkest_color = base_color

            # Create a lighter color variant if the base is dark, or darker if base is light
            if base_lightness < 0.5:
                # Base is dark, create lighter variant
                second_brightest_color = "#CCCCCC"
            else:
                # Base is light, create darker variant
                second_darkest_color = "#333333"

    # Ensure the colors are different
    if second_darkest_color == second_brightest_color:
        # If they ended up the same, modify one of them
        if get_lightness(second_darkest_color) < 0.5:
            # It's a dark color, make the bright one lighter
            second_brightest_color = "#CCCCCC"
        else:
            # It's a light color, make the dark one darker
            second_darkest_color = "#333333"

    # Base size for the outer circle
    base_outer_radius = width * 0.023

    # Randomize size by ±10%
    size_variation = base_outer_radius * 0.1
    outer_radius = base_outer_radius + random.uniform(-size_variation, size_variation)

    # Define radii for inner circles based on outer radius
    middle_radius = outer_radius * 0.80
    inner_radius = middle_radius * 0.65

    # Calculate the maximum crop margin based on the image processing (5% of dimensions)
    # Add 20% extra margin for safety
    crop_margin_w = int(width * 0.05 * 1.2)
    crop_margin_h = int(height * 0.05 * 1.2)

    # Calculate center point based on the outer radius to ensure the entire circle stays visible
    safe_offset = outer_radius + max(crop_margin_w, crop_margin_h)

    # Choose a random corner (0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right)
    if corner is None:
        corner = random.randint(0, 3)

    # Position the circle in the chosen corner, accounting for crop margin
    if corner == 0:  # Top-left
        center_x = safe_offset
        center_y = safe_offset
    elif corner == 1:  # Top-right
        center_x = width - safe_offset
        center_y = safe_offset
    elif corner == 2:  # Bottom-left
        center_x = safe_offset
        center_y = height - safe_offset
    else:  # Bottom-right
        center_x = width - safe_offset
        center_y = height - safe_offset

    # Add a small random offset (±10% of safe_offset) to make positioning less predictable
    random_offset = safe_offset * 0.1
    center_x += random.uniform(-random_offset, random_offset)
    center_y += random.uniform(-random_offset, random_offset)

    # Round to 1 decimal place to keep file size down
    outer_radius = round(outer_radius, 1)
    middle_radius = round(middle_radius, 1)
    inner_radius = round(inner_radius, 1)
    center_x = round(center_x, 1)
    center_y = round(center_y, 1)

    # Create the nested circles
    outer_circle = f'<circle cx="{center_x}" cy="{center_y}" r="{outer_radius}" fill="{second_darkest_color}"/>'
    middle_circle = f'<circle cx="{center_x}" cy="{center_y}" r="{middle_radius}" fill="{second_brightest_color}"/>'
    inner_circle = f'<circle cx="{center_x}" cy="{center_y}" r="{inner_radius}" fill="{second_darkest_color}"/>'

    # Create a group element that contains all three circles
    #group_element = f'<g>{outer_circle}{middle_circle}{inner_circle}</g>'
    group_element = f'{outer_circle}{middle_circle}{inner_circle}'

    # Insert the group element just before the closing SVG tag
    modified_svg = svg_code.replace("</svg>", f"{group_element}</svg>")

    # Calculate and add a comment with the byte size information
    outer_bytes = len(outer_circle.encode('utf-8'))
    middle_bytes = len(middle_circle.encode('utf-8'))
    inner_bytes = len(inner_circle.encode('utf-8'))
    total_bytes = outer_bytes + middle_bytes + inner_bytes

    corner_names = ["top-left", "top-right", "bottom-left", "bottom-right"]
    byte_info = f'<!-- Circle bytes: outer={outer_bytes}, middle={middle_bytes}, ' \
                f'inner={inner_bytes}, total={total_bytes}, ' \
                f'colors: dark={second_darkest_color}, light={second_brightest_color}, ' \
                f'position: {corner_names[corner]} -->'

    #modified_svg = modified_svg.replace("</svg>", f"{byte_info}</svg>")

    return modified_svg


def _rgb_to_hex(m: re.Match) -> str:
    r, g, b = map(int, m.groups())
    return f'#{r:02x}{g:02x}{b:02x}'


def _strip_number(num: str) -> str:
    if "." in num:
        num = num.rstrip("0").rstrip(".")
    if num.startswith("-0."):
        num = "-." + num[3:]
    elif num.startswith("0.") and len(num) > 2:
        num = num[1:]
    return num or "0"


def _scale_and_fmt(v: float, scale: float, d: int) -> str:
    return _strip_number(f"{round(v * scale, d):.{d}f}")


def _extract_canvas_size(svg: str):
    m = re.search(r'<svg[^>]*\bwidth="([0-9.]+)"[^>]*\bheight="([0-9.]+)"', svg)
    if not m:
        raise ValueError("width/height attributes not found in <svg> tag")
    return float(m.group(1)), float(m.group(2))


def compress_svg(svg: str, *, out_size: int = 960, d_point: int = 0, d_opa: int = 2) -> str:
    # ── 0. get original canvas size
    orig_w, orig_h = _extract_canvas_size(svg)
    scale = out_size / max(orig_w, orig_h)

    # ── 1. rewrite header
    svg = re.sub(
        r'<svg[^>]*>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{out_size}" height="{out_size}" '
        f'viewBox="0 0 {out_size} {out_size}">', svg, count=1)

    # ── 2. strip unneeded tags/attrs
    svg = re.sub(r'<\?xml[^>]*\?>\s*', '', svg)
    svg = svg.replace('<g>', "").replace('</g>', "") \
            .replace('<defs/>', "").replace('version="1.1" ', "")

    # ── 3. rescale <rect>
    def fix_rect(m: re.Match) -> str:
        attrs = m.group(1)
        fill = re.search(r'fill="[^"]+"', attrs)
        x  = float(re.search(r'\bx="([^"]+)"', attrs).group(1))
        y  = float(re.search(r'\by="([^"]+)"', attrs).group(1))
        w  = float(re.search(r'\bwidth="([^"]+)"',  attrs).group(1))
        h  = float(re.search(r'\bheight="([^"]+)"', attrs).group(1))

        parts = ["<rect"]
        if x or scale != 1.0:
            parts.append(f'x="{_scale_and_fmt(x, scale, d_point)}"')
        if y or scale != 1.0:
            parts.append(f'y="{_scale_and_fmt(y, scale, d_point)}"')
        parts.append(f'width="{_scale_and_fmt(w, scale, d_point)}"')
        parts.append(f'height="{_scale_and_fmt(h, scale, d_point)}"')
        if fill:
            parts.append(fill.group(0))
        parts.append("/>")
        return " ".join(parts)

    svg = re.sub(r'<rect\s+([^>]*)/?>', fix_rect, svg, count=1)

    # ── 4. polygons → paths
    def polygon_to_path(m: re.Match) -> str:
        leading, pts, trailing = m.group(1), m.group(2), m.group(3)
        attrs = leading + trailing
        fill    = re.search(r'fill="[^"]+"', attrs)
        opacity = re.search(r'opacity="[^"]+"', attrs)

        coords = [_scale_and_fmt(float(v), scale, d_point) for v in pts.split()]
        d_attr = " ".join(
            ('M' if i == 0 else '') + coords[i] + " " + coords[i + 1]
            for i in range(0, len(coords), 2)
        ) + "Z"

        pieces = ['<path', f'd="{d_attr}"']
        if fill:
            pieces.append(fill.group(0))
        if opacity:
            pieces.append(opacity.group(0))
        pieces.append("/>")
        return " ".join(pieces)

    svg = re.sub(
        r'<polygon\s+([^>/]*?)\bpoints="([^"]+)"([^>/]*)/?>',
        polygon_to_path, svg)

    # ── 5. opacity: round & drop when == 1
    def fix_opacity(m: re.Match) -> str:
        val = round(float(m.group(1)), d_opa)
        if val == 1:
            return ""                          # remove attribute entirely
        return f' opacity="{_strip_number(f"{val:.{d_opa}f}")}"'

    svg = re.sub(r'\s*opacity="([^"]+)"', fix_opacity, svg)

    # ── 6. rgb() → #rrggbb
    svg = re.sub(r'rgb\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\)', _rgb_to_hex, svg)

    # ── 7. cleanup
    svg = re.sub(r'(\d+)\.0\b', r'\1', svg)
    svg = re.sub(r'\s+', " ", svg).strip()
    svg = re.sub(r'>\s+<', "><", svg)
    svg = re.sub(r'\s+/>', "/>", svg)

    return svg


def optimize_svg(svg, image, n_iter=100, point_lr=2.0, color_lr=0.05,
                 warmup_steps=0, cosine_schedule=False, loss_fn=None,
                 optimizer=torch.optim.Adam, return_best=False, max_color_deviation=None,
                 color_decay=0.0, point_decay=0.0):
  target = utils.image_to_tensor([image], dtype=torch.float32)

  render = pydiffvg.RenderFunction.apply

  root = etree.fromstring(svg)
  canvas_width, canvas_height, shapes, shape_groups = pydiffvg.parse_scene(root)
  scene_args = pydiffvg.RenderFunction.serialize_scene(
      canvas_width, canvas_height, shapes, shape_groups)

  points_vars = []
  initial_points = {}
  for path in shapes:
      if not isinstance(path, pydiffvg.Rect):
          path.points.requires_grad = True
          points_vars.append(path.points)
          initial_points[path.points.data_ptr()] = path.points.data.clone()

  color_vars = {}
  initial_colors = {}
  for group in shape_groups:
      group.fill_color.requires_grad = True
      color_vars[group.fill_color.data_ptr()] = group.fill_color
      initial_colors[group.fill_color.data_ptr()] = group.fill_color.data.clone()
  color_vars = list(color_vars.values())

  points_optim = optimizer(points_vars, lr=point_lr)
  color_optim = optimizer(color_vars, lr=color_lr)
  if cosine_schedule:
    points_sched = get_cosine_schedule_with_warmup(
      optimizer=points_optim,
      num_warmup_steps=warmup_steps,
      num_training_steps=n_iter
    )
    color_sched = get_cosine_schedule_with_warmup(
        optimizer=color_optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=n_iter
    )
  else:
    points_sched = get_constant_schedule_with_warmup(points_optim, warmup_steps)
    color_sched = get_constant_schedule_with_warmup(color_optim, warmup_steps)

  best_svg = None
  lowest_loss = 1000
  losses = []
  for t in range(n_iter):
      points_optim.zero_grad()
      color_optim.zero_grad()
      # Forward pass: render the image.
      scene_args = pydiffvg.RenderFunction.serialize_scene(\
          canvas_width, canvas_height, shapes, shape_groups)
      img = render(canvas_width, # width
                    canvas_height, # height
                    2,   # num_samples_x
                    2,   # num_samples_y
                    0,   # seed
                    None, # bg
                    *scene_args)
      alpha = img[:, :, 3:4]
      img = alpha * img[:, :, :3] + (1 - alpha)
      img = img[:, :, :3].unsqueeze(0).permute(0, 3, 1, 2)

      if loss_fn is not None:
          loss = loss_fn(img, target)
          losses.append(loss.item())
          if t % 2 == 0 or t == n_iter - 1:
            print(f'step={t}, loss={loss.item()}')
          # print(f'step={t}, loss={loss.item()}')
          if loss < lowest_loss and return_best:
            lowest_loss = loss
            with tempfile.NamedTemporaryFile('r+', delete=False, suffix=".svg") as tmpfile:
                pydiffvg.save_svg(tmpfile.name, canvas_width, canvas_height, shapes, shape_groups)
                tmpfile.seek(0)
                best_svg = tmpfile.read()
      else:
          loss = torch.abs(img - target).mean()
      loss.backward()

      points_optim.step()
      color_optim.step()
      points_sched.step()
      color_sched.step()

      for i, group in enumerate(shape_groups):
          initial_color = initial_colors[group.fill_color.data_ptr()]
          if color_decay > 0:
              group.fill_color.data = color_decay * group.fill_color.data + (1 - color_decay) * initial_color
          group.fill_color.data.clamp_(0.0, 1.0)
          if i == 0:
              group.fill_color.data[-1] = 1.0
          if max_color_deviation is not None:
              min_color = torch.clamp(initial_color - max_color_deviation, 0.0, 1.0)
              max_color = torch.clamp(initial_color + max_color_deviation, 0.0, 1.0)
              group.fill_color.data.clamp_(min_color, max_color)

      for point in points_vars:
          initial_point = initial_points[point.data_ptr()]
          if point_decay > 0:
              point.data = point_decay * point.data + (1 - point_decay) * initial_point
          point.data.clamp_(0.0, canvas_height)

  if best_svg is not None:
    return best_svg
  with tempfile.NamedTemporaryFile('r+', delete=False, suffix=".svg") as tmpfile:
      pydiffvg.save_svg(tmpfile.name, canvas_width, canvas_height, shapes, shape_groups)
      tmpfile.seek(0)
      return tmpfile.read()


def make_svg(image, refine_iters=40):
  svg, _ = make_rough_svg(image, 9900)
  svg = optimize_svg(svg, image, n_iter=40)
  return compress_svg(svg)
