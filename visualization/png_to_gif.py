#!/usr/bin/env python3
"""
Standalone PNG to GIF converter - No browser required!

This script converts a sequence of PNG images to an animated GIF.
Perfect for converting existing PNG visualizations to GIF without needing Chrome/Chromium.
"""

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
    
    # Find all PNG files matching the pattern
    png_files = sorted(png_dir.glob('visualization_step_*.png'))
    if not png_files:
        # Try any PNG files
        png_files = sorted(png_dir.glob('*.png'))
    
    if not png_files:
        raise ValueError(f"No PNG files found in {png_dir}")
    
    # Determine output path
    if output_gif is None:
        output_gif = os.path.join(png_dir.parent, f"{png_dir.name}.gif")
    
    print(f"Found {len(png_files)} PNG files")
    print(f"Converting to GIF: {output_gif}")
    
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
        print(f"\n✓ GIF saved to {output_gif}")
        print(f"  - {len(images)} frames")
        print(f"  - Duration: {duration}ms per frame")
        print(f"  - Total time: {len(images) * duration / 1000:.2f}s")
        print(f"  - Loop: {'infinite' if loop == 0 else f'{loop} times'}")
    except Exception as e:
        raise RuntimeError(f"Error creating GIF: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert PNG sequence to animated GIF (No browser required!)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert PNG directory to GIF
  python png_to_gif.py png/my_visualization
  
  # Custom duration and output
  python png_to_gif.py png/my_visualization --output my_animation.gif --duration 200
  
  # Loop 3 times instead of infinite
  python png_to_gif.py png/my_visualization --loop 3
        """
    )
    parser.add_argument('png_dir', type=str, 
                       help='Directory containing PNG files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output GIF file path (default: {png_dir}.gif)')
    parser.add_argument('--duration', type=int, default=100,
                       help='Frame duration in milliseconds (default: 100)')
    parser.add_argument('--loop', type=int, default=0,
                       help='Loop count (0 = infinite, default: 0)')
    
    args = parser.parse_args()
    
    try:
        convert_png_to_gif(
            args.png_dir,
            output_gif=args.output,
            duration=args.duration,
            loop=args.loop
        )
    except Exception as e:
        print(f"\nError: {e}")
        exit(1)
