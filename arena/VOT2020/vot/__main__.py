import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from vot.cli import main

if __name__ == '__main__':
    main()

# evaluate --workspace /home/zxy/Desktop/AlphaRefine/analysis/VOT2020 AlphaRef
# analysis --workspace /home/zxy/Desktop/AlphaRefine/analysis/VOT2020 AlphaRef --output json
