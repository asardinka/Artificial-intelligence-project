from __future__ import annotations

from pathlib import Path
from typing import cast

import typer

from src.tumor_app.infer import Plane, predict_from_path

app = typer.Typer(help="CLI инференса опухолей (те же веса, что и у HTTP API)")


@app.command()
def predict(
    image: Path = typer.Argument(..., exists=True, readable=True, help="Путь к изображению"),
    plane: str = typer.Option("ax", help="Плоскость: ax, sa, co"),
    out_dir: Path = typer.Option(Path("out"), help="Каталог для original.png и overlay.png"),
) -> None:
    pl = plane.lower().strip()
    if pl not in ("ax", "sa", "co"):
        raise typer.BadParameter("plane должен быть ax, sa или co")

    out_dir.mkdir(parents=True, exist_ok=True)
    r = predict_from_path(image, cast(Plane, pl))
    (out_dir / "original.png").write_bytes(r.original_png)
    (out_dir / "overlay.png").write_bytes(r.overlay_png)
    typer.echo(f"class={r.predicted_class}  plane={r.plane}")
    typer.echo(f"saved: {out_dir / 'original.png'}, {out_dir / 'overlay.png'}")


if __name__ == "__main__":
    app()
