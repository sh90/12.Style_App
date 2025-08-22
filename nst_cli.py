import argparse
from PIL import Image
from utils import pick_device, load_image
from nst import run_style_transfer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True, help="Path to content image")
    ap.add_argument("--style", required=True, help="Path to style image")
    ap.add_argument("--out", default="stylized.jpg", help="Output path")
    ap.add_argument("--size", type=int, default=384, help="Max shorter side")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--style_weight", type=float, default=1e6)
    ap.add_argument("--content_weight", type=float, default=1.0)
    ap.add_argument("--tv_weight", type=float, default=1e-4)
    args = ap.parse_args()

    device = pick_device()
    content = load_image(args.content, max_size=args.size)
    style = load_image(args.style, max_size=args.size)

    out = run_style_transfer(
        content, style,
        num_steps=args.steps,
        style_weight=args.style_weight,
        content_weight=args.content_weight,
        tv_weight=args.tv_weight,
        device=device,
    )
    out.save(args.out, quality=95)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
