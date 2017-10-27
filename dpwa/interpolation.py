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


class LinearInterpolation(InterpolationBase):
    def __init__(self, start, end, target):
        self._start = start
        self._end = end
        self._target = target

    def __call__(self, clock, peer_clock, loss, peer_loss):
        # Interpolate based on the client clock value linearly from s to e
        if clock > self._target:
            return self._end

        m = (self._end - self._start) / self._target
        return self._start + clock * m


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


class LossDivergenceInterpolation(InterpolationBase):
    def __init__(self, target_divergence_loss):
        self.target_divergence_loss = target_divergence_loss

    def __call__(self, clock, peer_clock, loss, peer_loss):
        loss2 = loss ** 2
        peer_loss2 = peer_loss ** 2
        k = loss2 / (loss2 + peer_loss2)
        return k * min(loss / self.target_divergence_loss, 1.0)
