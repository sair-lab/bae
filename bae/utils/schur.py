import pypose as pp

class TrustRegion(pp.optim.strategy.TrustRegion):
    def update(self, pg, last, loss, J, D, R, *args, **kwargs):
        J = [i.to_sparse_coo() for i in J]
        JD = None
        for i in range(len(D)):
            if JD is None:
                JD = J[i] @ D[i]
            else:
                JD += J[i] @ D[i]
        JD = JD[..., None]
        quality = (last - loss) / -((JD).mT @ (2 * R.view_as(JD) + JD)).squeeze()
        pg['radius'] = 1. / pg['damping']
        if quality > pg['high']:
            pg['radius'] = pg['up'] * pg['radius']
            pg['down'] = self.down
        elif quality > pg['low']:
            pg['radius'] = pg['radius']
            pg['down'] = self.down
        else:
            pg['radius'] = pg['radius'] * pg['down']
            pg['down'] = pg['down'] * pg['factor']
        pg['down'] = max(self.min, min(pg['down'], self.max))
        pg['radius'] = max(self.min, min(pg['radius'], self.max))
        pg['damping'] = 1. / pg['radius']

class Adaptive(pp.optim.strategy.Adaptive):
    def update(self, pg, last, loss, J, D, R, *args, **kwargs):
        J = [i.to_sparse_coo() for i in J]
        JD = None
        for i in range(len(D)):
            if JD is None:
                JD = J[i] @ D[i]
            else:
                JD += J[i] @ D[i]
        JD = JD[..., None]
        quality = (last - loss) / -((JD).mT @ (2 * R.view_as(JD) + JD)).squeeze()
        if quality > pg['high']:
            pg['damping'] = pg['damping'] * pg['down']
        elif quality > pg['low']:
            pg['damping'] = pg['damping']
        else:
            pg['damping'] = pg['damping'] * pg['up']
        pg['damping'] = max(self.min, min(pg['damping'], self.max))
