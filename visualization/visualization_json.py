import json
import os
from typing import List, Dict, Optional
from pathlib import Path
import sys

# Add parent directory to path to import load_tokenizer
sys.path.insert(0, str(Path(__file__).parent.parent))
from load_model import load_tokenizer

# Import for GIF conversion
try:
    from PIL import Image
    from html2image import Html2Image
    HAS_GIF_SUPPORT = True
except ImportError as e:
    HAS_GIF_SUPPORT = False
    print(f"Warning: GIF conversion requires PIL and html2image: {e}")

def convert_png_to_gif(png_dir: str, output_gif: str = None, duration: int = 100, loop: int = 0):
    """Convert PNG sequence to animated GIF."""
    if not HAS_GIF_SUPPORT:
        raise ImportError("PIL (Pillow) is required for GIF conversion. Install with: pip install pillow")
    
    png_dir = Path(png_dir)
    if not png_dir.exists():
        raise FileNotFoundError(f"PNG directory not found: {png_dir}")
    
    png_files = sorted(png_dir.glob('visualization_step_*.png'))
    if not png_files:
        raise ValueError(f"No PNG files found in {png_dir}")
    
    if output_gif is None:
        output_gif = os.path.join(png_dir.parent, f"{png_dir.name}.gif")
    
    print(f"Converting {len(png_files)} PNG files to GIF...")
    
    images = []
    for png_file in png_files:
        try:
            img = Image.open(png_file)
            images.append(img.copy())
        except Exception as e:
            print(f"Warning: Could not load {png_file}: {e}")
    
    if not images:
        raise ValueError("No valid images found")
    
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop
    )
    print(f"GIF saved to {output_gif}")

def convert_html_to_png_for_gif(html_dir: str, output_dir: str, size: tuple = (1200, 800), 
                                 start_step: int = 0, end_step: int = None):
    """Convert HTML files to PNG for GIF creation."""
    if not HAS_GIF_SUPPORT:
        raise ImportError("html2image is required. Install with: pip install html2image")
    
    html_dir = Path(html_dir)
    if not html_dir.exists():
        raise FileNotFoundError(f"HTML directory not found: {html_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if end_step is None:
        step_files = sorted(html_dir.glob('visualization_step_*.html'))
        if step_files:
            step_nums = []
            for f in step_files:
                try:
                    step_num = int(f.stem.split('_')[-1])
                    step_nums.append(step_num)
                except:
                    pass
            end_step = max(step_nums) if step_nums else 0
        else:
            end_step = 0
    
    hti = Html2Image()
    hti.browser.use_new_headless = None
    hti.output_path = output_dir
    
    for step in range(start_step, end_step + 1):
        html_path = html_dir / f'visualization_step_{step}.html'
        if not html_path.exists():
            continue
        
        if (step - start_step + 1) % 10 == 0:
            print(f"  Converting step {step} to PNG...")
        
        try:
            hti.screenshot(
                url=str(html_path),
                save_as=f'visualization_step_{step}.png',
                size=size
            )
        except Exception as e:
            print(f"Error converting step {step}: {e}")

MASK_ID = 126336  # Mask token ID

def load_trace_json(json_path: str) -> Dict:
    """Load trace JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def reconstruct_tokens_at_step(trace: Dict, step: int, tokenizer) -> List[str]:
    """
    Reconstruct the token sequence at a specific step.
    At each step:
    - Positions that are still masked (mask_index[i] == True) show predictions
    - Positions that are unmasked (mask_index[i] == False) show the final token
    """
    prompt = trace['prompt'][0]  # Get first batch
    final_sequence = trace['final_sequence'][0]  # Get first batch
    
    # Get all step traces
    step_traces = trace.get('step_traces', [])
    
    # Initialize mask_state
    mask_state = []
    
    if step >= len(step_traces):
        # Use final sequence if step is beyond available traces
        token_ids = final_sequence
        mask_state = [False] * len(token_ids)
    else:
        step_trace = step_traces[step]
        mask_index = step_trace.get('mask_index', [])
        predictions = step_trace.get('predictions', None)
        
        mask_state = mask_index[0] if mask_index else []
        
        # Build token sequence: use predictions where masked, final_sequence where unmasked
        token_ids = []
        preds = predictions[0] if predictions else None
        
        seq_len = len(final_sequence)
        for i in range(seq_len):
            if i < len(mask_state) and mask_state[i]:
                # Still masked: show prediction if available, otherwise show mask token
                if preds and i < len(preds):
                    token_ids.append(preds[i])
                else:
                    token_ids.append(MASK_ID)
            else:
                # Unmasked: show final token
                token_ids.append(final_sequence[i])
    
    # Convert token IDs to text tokens
    tokens = []
    for i, token_id in enumerate(token_ids):
        # Check if this position is still masked
        is_masked = (i < len(mask_state) and mask_state[i])
        
        if is_masked:
            # Still masked at this step - show as mask
            tokens.append('<|mdm_mask|>')
        elif token_id == MASK_ID:
            tokens.append('<|mdm_mask|>')
        else:
            try:
                # Convert token ID to token text
                token_text = tokenizer.convert_ids_to_tokens([token_id])[0]
                tokens.append(token_text)
            except:
                tokens.append(f"<{token_id}>")
    
    return tokens

def track_unmasking_order(trace: Dict) -> Dict[int, int]:
    """
    Track when each position was first unmasked.
    Returns dict mapping position -> step when unmasked.
    """
    unmasking_order = {}
    step_traces = trace.get('step_traces', [])
    
    # Track mask state across steps
    prev_mask_state = None
    
    for step_idx, step_trace in enumerate(step_traces):
        mask_index = step_trace.get('mask_index', [])
        if prev_mask_state is None:
            prev_mask_state = mask_index[0] if mask_index else []
        
        current_mask_state = mask_index[0] if mask_index else []
        
        # Find positions that changed from masked (True) to unmasked (False)
        for pos in range(min(len(prev_mask_state), len(current_mask_state))):
            if prev_mask_state[pos] and not current_mask_state[pos]:
                # This position was just unmasked
                if pos not in unmasking_order:
                    unmasking_order[pos] = step_idx
        
        prev_mask_state = current_mask_state
    
    return unmasking_order

def generate_background_color(step: int, max_step: int) -> str:
    """Generate gradient color based on step."""
    color_stops = [
        (240, 248, 255), (209, 226, 241), (176, 202, 224), (143, 179, 207),
        (110, 156, 191), (77, 133, 175), (44, 110, 159), (12, 55, 112)
    ]
    
    if max_step == 0:
        color_index = 0
    else:
        color_index = min(step * 6 // max_step, 6)
    ratio = (step % 2) / 2 if max_step > 0 else 0
    
    start = color_stops[color_index]
    end = color_stops[min(color_index + 1, 7)]
    
    r = int(start[0] + (end[0] - start[0]) * ratio)
    g = int(start[1] + (end[1] - start[1]) * ratio)
    b = int(start[2] + (end[2] - start[2]) * ratio)
    
    return f"#{r:02x}{g:02x}{b:02x}"

def generate_step_visualization(current_step: int, current_tokens: List[str], 
                               token_steps: Dict[int, int], max_step: int) -> str:
    """Generate visualization HTML for specific step."""
    html = []
    
    for idx, token in enumerate(current_tokens):
        style = [
            "color: #000000",
            "padding: 6px 8px",
            "margin: 3px",
            "border-radius: 6px",
            "display: inline-block",
            "font-weight: 600",
            "font-size: 16px",
            "font-family: 'Segoe UI', sans-serif",
            "box-shadow: 0 3px 6px rgba(12,55,112,0.15)",
            "transition: all 0.2s ease",
            "position: relative",
            "width: 120px",
            "min-width: 120px",
            "text-align: center",
            "white-space: nowrap",
            "overflow: hidden",
            "text-overflow: ellipsis",
            "box-sizing: border-box"
        ]

        if token == '<|mdm_mask|>':
            style.extend([
                "background: #f8fafc",
                "border: 2px solid #ffffff",
                "font-weight: 800",
                "text-transform: uppercase",
                "padding: 4px 6px"
            ])
            display_text = "Mask"
        else:
            # Get the step when this token was unmasked
            unmask_step = token_steps.get(idx, max_step + 1)
            bg_color = generate_background_color(unmask_step, max_step)
            style.append(f"background-color: {bg_color}")
            # Clean up token text for display
            display_text = token.replace('Ġ', ' ').strip()
            if not display_text or display_text == '':
                display_text = "␣"
            
        html.append(f'<span style="{"; ".join(style)}">{display_text}</span>')
    
    return '\n'.join(html)

def main(json_path: str, model_dir: str = "./models/llada", output_dir: Optional[str] = None,
         convert_to_gif: bool = False, gif_duration: int = 100, gif_loop: int = 0,
         png_size: tuple = (1200, 800)):
    """
    Main function to generate visualizations from JSON trace file.
    
    Args:
        json_path: Path to JSON trace file
        model_dir: Path to model directory (for tokenizer)
        output_dir: Output directory for HTML files (default: html/{filename})
    """
    print(f"Loading trace from {json_path}...")
    trace = load_trace_json(json_path)
    
    print(f"Loading tokenizer from {model_dir}...")
    tokenizer = load_tokenizer(model_dir)
    
    # Determine number of steps
    step_traces = trace.get('step_traces', [])
    num_steps = len(step_traces)
    if num_steps == 0:
        print("Warning: No step traces found in JSON file")
        return
    
    print(f"Found {num_steps} steps")
    
    # Track unmasking order
    print("Tracking unmasking order...")
    unmasking_order = track_unmasking_order(trace)
    
    # Determine output directory
    if output_dir is None:
        json_filename = Path(json_path).stem
        output_dir = f"html/{json_filename}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sequence length
    final_sequence = trace['final_sequence'][0]
    seq_length = len(final_sequence)
    
    print(f"Generating visualizations for {num_steps} steps (sequence length: {seq_length})...")
    
    # Generate visualization for each step
    for step in range(num_steps):
        if (step + 1) % 10 == 0:
            print(f"  Processing step {step + 1}/{num_steps}...")
        
        # Reconstruct tokens at this step
        tokens = reconstruct_tokens_at_step(trace, step, tokenizer)
        
        # Truncate or pad to sequence length
        if len(tokens) > seq_length:
            tokens = tokens[:seq_length]
        elif len(tokens) < seq_length:
            tokens.extend(['<|mdm_mask|>'] * (seq_length - len(tokens)))
        
        # Generate HTML
        html_content = generate_step_visualization(
            step, tokens, unmasking_order, num_steps - 1
        )
        
        # Create legend
        example_steps = [0, num_steps // 4, num_steps // 2, 3 * num_steps // 4, num_steps - 1]
        example_steps = [s for s in example_steps if s < num_steps]
        example_colors = [generate_background_color(s, num_steps - 1) for s in example_steps]
        legend_html = ''.join(
            f'<div style="background-color: {color}; color: black; padding: 10px; border-radius: 5px; min-width: 80px; text-align: center;">Step {s}</div>'
            for s, color in zip(example_steps, example_colors)
        )
        
        # Write HTML file
        html_file = os.path.join(output_dir, f'visualization_step_{step}.html')
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(f"""<html>
<head>
    <title>Step {step} Visualization</title>
    <style>
        body {{ 
            padding: 40px;
            background: #f8fafc;
            font-family: 'Segoe UI', sans-serif;
        }}
        .legend {{
            display: flex;
            gap: 15px;
            margin: 20px 0;
        }}
        .legend div {{
            color: white;
        }}
        .token:hover {{
            transform: translateY(-2px);
        }}
    </style>
</head>
<body>
    <div style="max-width: 1400px; margin: auto;">
        <h2>Generation Step {step}</h2>
        <div>{html_content}</div>
        
        <h3>Color Legend</h3>
        <div class="legend">{legend_html}</div>
    </div>
</body>
</html>""")
    
    print(f"\nVisualizations saved to {output_dir}/")
    print(f"Generated {num_steps} HTML files")
    
    # Convert to GIF if requested
    if convert_to_gif:
        if not HAS_GIF_SUPPORT:
            print("\nError: GIF conversion requires 'pillow' package.")
            print("Install with: pip install pillow")
            print(f"\nYou can manually convert PNG to GIF later using:")
            print(f"  python html_to_png_json.py png/{Path(output_dir).name} --png_to_gif_only")
            return
        
        print("\n" + "="*50)
        print("Converting HTML to PNG, then to GIF...")
        print("="*50)
        
        # First convert HTML to PNG
        png_output_dir = output_dir.replace('html/', 'png/')
        html_to_png_success = False
        
        try:
            convert_html_to_png_for_gif(
                output_dir,
                png_output_dir,
                size=png_size,
                start_step=0,
                end_step=num_steps - 1
            )
            html_to_png_success = True
        except Exception as e:
            error_msg = str(e).lower()
            if 'chrome' in error_msg or 'chromium' in error_msg or 'browser' in error_msg:
                print("\n" + "="*60)
                print("WARNING: Chrome/Chromium not found - skipping HTML to PNG conversion")
                print("="*60)
                print("\nHTML files have been generated, but PNG conversion requires Chrome/Chromium.")
                print("\nOptions:")
                print("1. Install Chrome/Chromium and run:")
                print(f"   python html_to_png_json.py {output_dir} --convert_to_gif")
                print("\n2. If you already have PNG files, convert them to GIF:")
                print(f"   python html_to_png_json.py png/{Path(output_dir).name} --png_to_gif_only")
                print("\n3. Install Chrome/Chromium:")
                print("   - Ubuntu/Debian: sudo apt-get install chromium-browser")
                print("   - Or download from: https://www.google.com/chrome/")
                print("="*60)
            else:
                print(f"Error during HTML to PNG conversion: {e}")
        
        # Try to convert PNG to GIF if PNG conversion succeeded or PNGs already exist
        if html_to_png_success or Path(png_output_dir).exists():
            try:
                gif_output = os.path.join(Path(png_output_dir).parent, f"{Path(output_dir).name}.gif")
                convert_png_to_gif(
                    png_output_dir,
                    output_gif=gif_output,
                    duration=gif_duration,
                    loop=gif_loop
                )
                print(f"\n✓ GIF saved to {gif_output}")
            except Exception as e:
                print(f"Error during PNG to GIF conversion: {e}")
                print(f"\nYou can manually convert PNG to GIF using:")
                print(f"  python html_to_png_json.py {png_output_dir} --png_to_gif_only")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visualizations from JSON trace files')
    parser.add_argument('json_path', type=str, help='Path to JSON trace file')
    parser.add_argument('--model_dir', type=str, default='./models/llada',
                       help='Path to model directory (default: ./models/llada)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for HTML files (default: html/{filename})')
    parser.add_argument('--convert_to_gif', action='store_true',
                       help='Convert HTML visualizations to animated GIF (requires html2image)')
    parser.add_argument('--gif_duration', type=int, default=100,
                       help='GIF frame duration in milliseconds (default: 100)')
    parser.add_argument('--gif_loop', type=int, default=0,
                       help='GIF loop count (0 = infinite, default: 0)')
    parser.add_argument('--png_width', type=int, default=1200,
                       help='PNG screenshot width (default: 1200)')
    parser.add_argument('--png_height', type=int, default=800,
                       help='PNG screenshot height (default: 800)')
    
    args = parser.parse_args()
    
    main(
        args.json_path,
        args.model_dir,
        args.output_dir,
        convert_to_gif=args.convert_to_gif,
        gif_duration=args.gif_duration,
        gif_loop=args.gif_loop,
        png_size=(args.png_width, args.png_height)
    )
