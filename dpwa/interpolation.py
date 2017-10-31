"""Factor interpolation methods."""

class InterpolationBase:
    def __call__(self, clock, peer_clock, loss, peer_loss):
        raise NotImplementedError


class ConstantInterpolation(InterpolationBase):
    def __init__(self, value):
        assert (value <= 1) and (value >= 0)
        self._value = value

    def __call__(self, clock, peer_clock, loss, peer_loss):
        # Simply return a fixed value
        return self._value


class ClockWeightedInterpolation(InterpolationBase):
    def __init__(self):
        pass

    def __call__(self, clock, peer_clock, loss, peer_loss):
        # Return a factor based on the clock value
        return peer_clock / (clock + peer_clock)


class LossInterpolation(InterpolationBase):
    def __init__(self):
        pass

    def __call__(self, clock, peer_clock, loss, peer_loss):
        k =  loss / (loss + peer_loss)
        return k
