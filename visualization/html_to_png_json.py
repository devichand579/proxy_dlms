from html2image import Html2Image
import os
import argparse
from pathlib import Path
from PIL import Image

def convert_png_to_gif(png_dir: str, output_gif: str = None, duration: int = 100, loop: int = 0):
    """
    Convert PNG sequence to animated GIF.
    
    Args:
        png_dir: Directory containing PNG files
        output_gif: Output GIF file path (default: {png_dir}.gif)
        duration: Frame duration in milliseconds (default: 100)
        loop: Number of loops (0 = infinite, default: 0)
    """
    png_dir = Path(png_dir)
    if not png_dir.exists():
        raise FileNotFoundError(f"PNG directory not found: {png_dir}")
    
    # Find all PNG files
    png_files = sorted(png_dir.glob('visualization_step_*.png'))
    if not png_files:
        raise ValueError(f"No PNG files found in {png_dir}")
    
    # Determine output path
    if output_gif is None:
        output_gif = os.path.join(png_dir.parent, f"{png_dir.name}.gif")
    
    print(f"Converting {len(png_files)} PNG files to GIF...")
    print(f"Output: {output_gif}")
    
    # Load images
    images = []
    for png_file in png_files:
        try:
            img = Image.open(png_file)
            images.append(img.copy())
        except Exception as e:
            print(f"Warning: Could not load {png_file}: {e}")
    
    if not images:
        raise ValueError("No valid images found")
    
    # Save as animated GIF
    try:
        images[0].save(
            output_gif,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop
        )
        print(f"GIF saved to {output_gif}")
        print(f"  - {len(images)} frames")
        print(f"  - Duration: {duration}ms per frame")
        print(f"  - Total time: {len(images) * duration / 1000:.2f}s")
    except Exception as e:
        raise RuntimeError(f"Error creating GIF: {e}")

def convert_html_to_png(html_dir: str, output_dir: str = None, size: tuple = (1200, 800), 
                        start_step: int = 0, end_step: int = None, convert_to_gif: bool = False,
                        gif_duration: int = 100, gif_loop: int = 0):
    """
    Convert HTML visualization files to PNG images.
    
    Args:
        html_dir: Directory containing HTML files
        output_dir: Output directory for PNG files (default: png/{html_dir_name})
        size: Screenshot size (width, height)
        start_step: First step to convert (default: 0)
        end_step: Last step to convert (default: None, auto-detect)
    """
    html_dir = Path(html_dir)
    if not html_dir.exists():
        raise FileNotFoundError(f"HTML directory not found: {html_dir}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.join('png', html_dir.name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Auto-detect number of steps if end_step not provided
    if end_step is None:
        step_files = sorted(html_dir.glob('visualization_step_*.html'))
        if step_files:
            # Extract step numbers
            step_nums = []
            for f in step_files:
                try:
                    step_num = int(f.stem.split('_')[-1])
                    step_nums.append(step_num)
                except:
                    pass
            if step_nums:
                end_step = max(step_nums)
            else:
                end_step = 0
        else:
            end_step = 0
    
    print(f"Converting HTML files from {html_dir} to PNG...")
    print(f"Steps: {start_step} to {end_step}")
    print(f"Output directory: {output_dir}")
    
    # Initialize Html2Image object
    hti = Html2Image()
    hti.browser.use_new_headless = None
    hti.output_path = output_dir
    
    # Convert each step
    for step in range(start_step, end_step + 1):
        html_path = html_dir / f'visualization_step_{step}.html'
        
        if not html_path.exists():
            print(f"Warning: {html_path} not found, skipping...")
            continue
        
        if (step - start_step + 1) % 10 == 0:
            print(f"  Processing step {step}...")
        
        # Generate screenshot
        try:
            hti.screenshot(
                url=str(html_path),
                save_as=f'visualization_step_{step}.png',
                size=size
            )
        except Exception as e:
            print(f"Error converting step {step}: {e}")
    
    print(f"\nPNG files saved to {output_dir}/")
    print(f"Converted {end_step - start_step + 1} steps")
    
    # Convert to GIF if requested
    if convert_to_gif:
        print("\nConverting PNG sequence to GIF...")
        convert_png_to_gif(output_dir, duration=gif_duration, loop=gif_loop)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert HTML visualizations to PNG images, or PNG to GIF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert HTML to PNG (requires Chrome/Chromium)
  python html_to_png_json.py html/my_visualization
  
  # Convert existing PNG directory to GIF (no browser needed)
  python html_to_png_json.py png/my_visualization --png_to_gif_only
  
  # Convert HTML to PNG then to GIF
  python html_to_png_json.py html/my_visualization --convert_to_gif
        """
    )
    parser.add_argument('input_dir', type=str, 
                       help='Directory containing HTML files (or PNG files if --png_to_gif_only)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for PNG files (default: png/{html_dir_name})')
    parser.add_argument('--width', type=int, default=1200, help='Screenshot width (default: 1200)')
    parser.add_argument('--height', type=int, default=800, help='Screenshot height (default: 800)')
    parser.add_argument('--start_step', type=int, default=0, help='First step to convert (default: 0)')
    parser.add_argument('--end_step', type=int, default=None, help='Last step to convert (default: auto-detect)')
    parser.add_argument('--convert_to_gif', action='store_true', 
                       help='Convert PNG sequence to animated GIF after PNG generation')
    parser.add_argument('--png_to_gif_only', action='store_true',
                       help='Skip HTML->PNG, only convert existing PNG files to GIF (no browser needed)')
    parser.add_argument('--gif_duration', type=int, default=100,
                       help='GIF frame duration in milliseconds (default: 100)')
    parser.add_argument('--gif_loop', type=int, default=0,
                       help='GIF loop count (0 = infinite, default: 0)')
    
    args = parser.parse_args()
    
    # If png_to_gif_only, just convert PNGs to GIF
    if args.png_to_gif_only:
        convert_png_to_gif(
            args.input_dir,
            duration=args.gif_duration,
            loop=args.gif_loop
        )
    else:
        # Try HTML to PNG conversion
        try:
            convert_html_to_png(
                args.input_dir,
                args.output_dir,
                size=(args.width, args.height),
                start_step=args.start_step,
                end_step=args.end_step,
                convert_to_gif=args.convert_to_gif,
                gif_duration=args.gif_duration,
                gif_loop=args.gif_loop
            )
        except Exception as e:
            error_msg = str(e).lower()
            if 'chrome' in error_msg or 'chromium' in error_msg or 'browser' in error_msg:
                print("\n" + "="*60)
                print("ERROR: Chrome/Chromium not found!")
                print("="*60)
                print("\nTo convert HTML to PNG, you need Chrome/Chromium installed.")
                print("\nOptions:")
                print("1. Install Chrome/Chromium:")
                print("   - Ubuntu/Debian: sudo apt-get install chromium-browser")
                print("   - Or download from: https://www.google.com/chrome/")
                print("\n2. Convert existing PNG files to GIF (no browser needed):")
                print(f"   python html_to_png_json.py png/{Path(args.input_dir).name} --png_to_gif_only")
                print("\n3. Use a different method to convert HTML to PNG first,")
                print("   then convert PNG to GIF using option 2.")
                print("="*60)
            else:
                raise
