Generate 2D voxelized images from CIFs

Usage: 
python GenMOFImage.py --cifdir CIFDIR --outdir OUTDIR --r R -res RES --grid GRID

optional arguments:
  -h, --help       show this help message and exit
  --cifdir CIFDIR  Directory containing CIF files (default: current directory)
  --outdir OUTDIR  Directory for generated images (default: same as --cifdir)
  --r R            Atomic radii: vdw (default), cov, or a float value in Å
  --res RES        Image resolution in pixels (default: 500)
  --grid GRID      Grid size in Å (default: 50)


Requirements:
numpy
matplotlib
ase