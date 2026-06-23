import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import io
    from pathlib import Path

    import fsspec
    import pyarrow.parquet as pq
    from PIL import Image as PILImage, ImageFile

    # Optional: tolerate slightly truncated images
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    OUT_DIR = Path("tests/fixtures/images/ai/grok")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    url = (
        "https://huggingface.co/datasets/ComplexDataLab/OpenFake/resolve/main/"
        "core/train-00000-of-00032-00000.parquet"
    )

    target_model = "grok-2-image-1212"
    limit = 400

    saved = 0
    bad = 0

    with fsspec.open(url, "rb") as f:
        pf = pq.ParquetFile(f)

        print("row groups:", pf.num_row_groups)
        print("schema:", pf.schema_arrow)

        for rg in range(pf.num_row_groups):
            if saved >= limit:
                break

            # Read only the model column first.
            model_table = pf.read_row_group(rg, columns=["model"])
            models = model_table.column("model").to_pylist()

            matching_offsets = [
                i for i, value in enumerate(models)
                if value == target_model
            ]

            if not matching_offsets:
                print(f"row group {rg}: no matches")
                continue

            print(f"row group {rg}: {len(matching_offsets)} matches")

            # Only now read image column for this row group.
            image_table = pf.read_row_group(rg, columns=["image"])
            images = image_table.column("image").to_pylist()

            for offset in matching_offsets:
                if saved >= limit:
                    break

                image_obj = images[offset]

                try:
                    # HF Image columns are usually structs like:
                    # {"bytes": b"...", "path": None}
                    if isinstance(image_obj, dict):
                        raw = image_obj.get("bytes")
                    elif isinstance(image_obj, bytes):
                        raw = image_obj
                    else:
                        raw = None

                    if not raw:
                        bad += 1
                        print(f"  bad image at rg={rg}, offset={offset}: no bytes")
                        continue

                    img = PILImage.open(io.BytesIO(raw))

                    # Force decode pixels, but avoid calling getexif()
                    img.load()

                    # Normalize for fixture use
                    img = img.convert("RGB")

                    out_path = OUT_DIR / f"grok_{saved:04d}.jpg"
                    img.save(out_path, quality=95)

                    saved += 1

                except Exception as e:
                    bad += 1
                    print(
                        f"  skipping rg={rg}, offset={offset}: "
                        f"{type(e).__name__}: {e}"
                    )

    print(f"done: saved={saved}, bad={bad}")
    return


if __name__ == "__main__":
    app.run()
