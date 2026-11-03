"""The picture area: the part of the coded frame a title's picture fills.

Many sources keep a full frame with a wider picture letterboxed inside it,
the black bars baked into the video (a 1920x1080 frame whose picture is
1920x804), or a narrower one pillarboxed (a 4:3 picture in a 16:9 frame).
The players fit the picture area to the screen rather than the frame. The
bars are measured once, here, and the catalog keeps the area; the package is
left as it is.

What the area is. The union of the picture over the whole title, every
scene: it is never smaller than the picture of a scene sampled, so a film
whose aspect changes (wider frames in places) keeps the largest, and a bar
counts only where it is black throughout. Cutting the picture is far worse
than missing a crop, so a side in doubt keeps the full frame.

  * sources.py: what is sampled (every shot of 3 s or more, where the
    keyframes allow), from a file opened once, and what that reads;
  * planes.py: a frame reduced to its lines' segment means, from its own
    Y, U and V planes;
  * rules.py: the area of the samples, counted as they come;
  * child.py, measure.py: a measurement in its own process, at a lower
    priority, stopped at its deadline.
"""

from .measure import METHOD, Measurement, measure
from .rules import Box

__all__ = ["METHOD", "Box", "Measurement", "measure"]
