# Pointillism Sunflower

A small Python animation that grows a sunflower from a bud, lets it bloom, and then lets it wilt and droop. The sun fades away into drifting pixels at the same time the flower dies. Everything is drawn as tiny dots (pointillism style) using NumPy and Pillow.

## What it does

1. A stem rises from the soil with a closed green bud.
2. The bud opens into a full sunflower that follows the sun across the sky.
3. The flower wilts, its petals fall to the ground, and the head bends over.
4. While this happens, the sun breaks apart into pixels and blows away.

## Requirements

- Python 3.8+
- `numpy`
- `Pillow`
- `tkinter` (for the live window; it comes with most Python installs)

```
pip install numpy pillow
```

## Usage

```
python sunflower.py          # live animation window
python sunflower.py --png    # save still frames and a stage strip
python sunflower.py --gif    # save an animated GIF (sunflower.gif)
```

Window controls:

| Key   | Action             |
|-------|--------------------|
| Space | Pause / resume     |
| R     | Restart            |
| S     | Save current frame |
| Esc   | Quit               |

## Settings

Edit the constants at the top of `sunflower.py`:

- `SEED`: changes the random look of the plant
- `TIME_SCALE`: speed of the animation (higher is slower)
- `LOOP`: set to `True` to replay forever
- `SOFTEN`: set to `0` for a harder, dotted look
- `KEYS`: the timeline table that controls growth, bloom, wilt and the sun
