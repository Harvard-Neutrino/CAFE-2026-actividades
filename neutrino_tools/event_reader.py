"""Lectura de los eventos de IceCube pre-extraidos (HDF5).

Los estudiantes solo usan `load_events(...)`, que devuelve un `EventSet`:

    events = load_events("neutrinos.h5")

    events.zenith            # arreglo numpy de todos los cenits (radianes)  -> histogramas
    up = events[events.zenith > np.radians(85)]   # filtrar con una mascara booleana
    event_display(up[0])     # un solo Event, listo para dibujar

El formato del HDF5 queda escondido aqui adentro; nunca se ve en la actividad.
"""

import numpy as np
import h5py


class Pulses:
    """Los DOMs que se encendieron en un evento (uno por DOM golpeado)."""

    def __init__(self, x, y, z, charge, time):
        self.x = x            # posicion del DOM [m]
        self.y = y
        self.z = z
        self.charge = charge  # carga total registrada [PE]
        self.time = time      # tiempo del primer pulso [ns]

    def __len__(self):
        return len(self.x)


class Geometry:
    """Posiciones de todos los DOMs del detector (para dibujar el detector)."""

    def __init__(self, x, y, z, string, om):
        self.x = x
        self.y = y
        self.z = z
        self.string = string
        self.om = om

    def string_lines(self):
        """Para cada string: (x, y, zmin, zmax) -> lineas verticales del detector."""
        lines = []
        for s in np.unique(self.string):
            m = self.string == s
            lines.append((self.x[m][0], self.y[m][0], self.z[m].min(), self.z[m].max()))
        return lines

    def hull_xy(self):
        """Poligono convexo (cerrado) de la huella (x, y) del detector."""
        pts = np.column_stack([self.x, self.y])
        pts = np.unique(pts, axis=0)
        pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        def build(seq):
            h = []
            for p in seq:
                while len(h) >= 2 and cross(h[-2], h[-1], p) <= 0:
                    h.pop()
                h.append(tuple(p))
            return h

        lower = build(pts)
        upper = build(pts[::-1])
        hull = lower[:-1] + upper[:-1]
        hull.append(hull[0])  # cerrar el anillo
        return np.array(hull)


class Event:
    """Un evento: la reconstruccion (direccion, calidad) + los DOMs golpeados."""

    def __init__(self, run, event, subevent, zenith, azimuth, vertex,
                 length, shape, rlogl, pulses, geometry=None):
        self.run = run
        self.event = event
        self.subevent = subevent
        self.zenith = zenith        # radianes (0 = hacia abajo, 90 = horizonte, 180 = hacia arriba)
        self.azimuth = azimuth      # radianes
        self.vertex = vertex        # (x, y, z) del vertice de la reconstruccion [m]
        self.length = length        # largo de la traza [m] (NaN = traza infinita)
        self.shape = shape
        self.rlogl = rlogl          # calidad del ajuste (menor = mejor)
        self.pulses = pulses        # Pulses
        self.geometry = geometry    # Geometry compartida (para el dibujo)

    @property
    def zenith_deg(self):
        return np.degrees(self.zenith)

    def __repr__(self):
        return (f"Event(run={self.run}, event={self.event}, "
                f"zenith={self.zenith_deg:.1f}deg, rlogl={self.rlogl:.2f}, "
                f"nDOM={len(self.pulses)})")


class EventSet:
    """Una coleccion de eventos que se comporta como un arreglo.

    - `events.zenith` / `.azimuth` / `.rlogl` -> arreglos numpy (para histogramas)
    - `events[mascara_booleana]` -> otro EventSet
    - `events[i]` -> un solo Event ; `for e in events: ...`
    """

    def __init__(self, events, geometry):
        self._events = list(events)
        self.geometry = geometry
        self.zenith = np.array([e.zenith for e in self._events])
        self.azimuth = np.array([e.azimuth for e in self._events])
        self.rlogl = np.array([e.rlogl for e in self._events])
        self.run = np.array([e.run for e in self._events])
        self.event = np.array([e.event for e in self._events])

    @property
    def zenith_deg(self):
        return np.degrees(self.zenith)

    def __len__(self):
        return len(self._events)

    def __iter__(self):
        return iter(self._events)

    def __repr__(self):
        return f"EventSet({len(self)} eventos)"

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self._events[int(key)]
        if isinstance(key, slice):
            return EventSet(self._events[key], self.geometry)
        key = np.asarray(key)
        if key.dtype == bool:
            sel = [e for e, m in zip(self._events, key) if m]
        else:
            sel = [self._events[int(i)] for i in key]
        return EventSet(sel, self.geometry)


def load_events(path):
    """Cargar el archivo HDF5 y devolver un `EventSet`."""
    with h5py.File(path, "r") as f:
        g = f["geometry"]
        geometry = Geometry(g["x"][:], g["y"][:], g["z"][:],
                            g["string"][:], g["om"][:])
        e = f["events"]
        run = e["run"][:]; ev = e["event"][:]; sub = e["subevent"][:]
        zen = e["zenith"][:]; azi = e["azimuth"][:]
        vx = e["vx"][:]; vy = e["vy"][:]; vz = e["vz"][:]
        length = e["length"][:]; shape = e["shape"][:]; rlogl = e["rlogl"][:]
        off = e["pulse_offset"][:]; cnt = e["pulse_count"][:]
        p = f["pulses"]
        px = p["x"][:]; py = p["y"][:]; pz = p["z"][:]
        pq = p["charge"][:]; pt = p["time"][:]

    events = []
    for i in range(len(zen)):
        s, n = int(off[i]), int(cnt[i])
        pulses = Pulses(px[s:s + n], py[s:s + n], pz[s:s + n],
                        pq[s:s + n], pt[s:s + n])
        events.append(Event(
            int(run[i]), int(ev[i]), int(sub[i]),
            float(zen[i]), float(azi[i]),
            (float(vx[i]), float(vy[i]), float(vz[i])),
            float(length[i]), int(shape[i]), float(rlogl[i]),
            pulses, geometry))
    return EventSet(events, geometry)
