# Contact sheet of screenshots, each labelled, for looking at many views at once.
#   python tools/scene3d/sheet.py <shots_dir> <out.jpg> [columns=3] [cell_width=960] [names...]
import sys, glob, os
import cv2, numpy as np
d, out = sys.argv[1], sys.argv[2]
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 3
cw = int(sys.argv[4]) if len(sys.argv) > 4 else 960
names = sys.argv[5:] or [os.path.basename(p)[:-4] for p in sorted(glob.glob(d + "/*.png"))]
cells = []
for n in names:
    im = cv2.imread(f"{d}/{n}.png")
    im = cv2.resize(im, (cw, int(im.shape[0] * cw / im.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.putText(im, n, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cells.append(im)
while len(cells) % cols:
    cells.append(np.zeros_like(cells[0]))
rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
