import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import Model
from darts.reservoirs.struct_reservoir import StructReservoir


class Capture(Model):
    """100x100x100 reservoir. The stock model file keeps its 1000-cell grid."""

    def set_reservoir(self):
        self.reservoir = StructReservoir(
            self.timer,
            nx=100,
            ny=100,
            nz=100,
            dx=1,
            dy=10,
            dz=10,
            permx=100,
            permy=100,
            permz=10,
            poro=0.3,
            depth=1000,
        )


def main():
    model = Capture()
    model.init()
    model.set_output(save_initial=False)
    model.run(0.015, save_well_data=False, save_reservoir_data=False)
    print("CAPTURE_DONE", flush=True)


if __name__ == "__main__":
    main()
