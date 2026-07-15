"""Visor 3D de eventos de IceCube, estilo steamshovel, en Plotly.

    event_display(event)

Dibuja el detector (tenue), los DOMs golpeados como burbujas (tamano = carga,
color = tiempo del primer pulso) y la traza reconstruida con una punta de flecha
que marca la direccion de propagacion.
"""

import numpy as np
import plotly.graph_objects as go


def _unit_sphere(n_lon=12, n_lat=8):
    """Malla (vertices, caras) de una esfera unitaria (UV) para instanciar por DOM."""
    lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    lat = np.linspace(0, np.pi, n_lat + 1)
    theta, phi = np.meshgrid(lat, lon, indexing="ij")
    verts = np.column_stack([
        (np.sin(theta) * np.cos(phi)).ravel(),
        (np.sin(theta) * np.sin(phi)).ravel(),
        (np.cos(theta)).ravel(),
    ])
    faces = []
    for j in range(n_lat):
        for i in range(n_lon):
            i2 = (i + 1) % n_lon
            a = j * n_lon + i; b = j * n_lon + i2
            c = (j + 1) * n_lon + i; d = (j + 1) * n_lon + i2
            faces.append((a, b, d)); faces.append((a, d, c))
    return verts, np.array(faces)


_SPHERE_V, _SPHERE_F = _unit_sphere()


def _direction_vector(zenith, azimuth):
    """Vector unitario de propagacion (hacia donde viaja la particula).

    El cenit/azimut nombran de donde VINO la particula, de ahi los signos menos.
    """
    sz, cz = np.sin(zenith), np.cos(zenith)
    sa, ca = np.sin(azimuth), np.cos(azimuth)
    return np.array([-sz * ca, -sz * sa, -cz])


def _nice_step(span, n=5):
    """Paso "redondo" (1/2/5 x 10^k) que da ~n intervalos a lo largo de `span`."""
    if span <= 0:
        return 1.0
    raw = span / n
    mag = 10.0 ** np.floor(np.log10(raw))
    r = raw / mag
    return (1.0 if r < 1.5 else 2.0 if r < 3.0 else 5.0 if r < 7.0 else 10.0) * mag


def _multiples(lo, hi, step):
    """Multiplos de `step` dentro de [lo, hi]."""
    return np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)


def _zaxis_traces(geo, ticklen=80.0, major=500.0, minor=100.0):
    """Regla vertical de z, fija en el lado derecho (+x) del detector.

    Reemplaza el eje z automatico de Plotly (que salta de esquina al rotar la
    camara) por una linea real con marcas y etiquetas, anclada al borde +x del
    arreglo, para que se quede siempre a la derecha.
    """
    x0 = float(geo.x.max()) + 120.0
    y0 = float(geo.y.mean())
    zmin, zmax = float(geo.z.min()), float(geo.z.max())
    traces = []

    # linea del eje
    traces.append(go.Scatter3d(
        x=[x0, x0], y=[y0, y0], z=[zmin, zmax], mode="lines",
        line=dict(color="black", width=2), hoverinfo="skip", showlegend=False))

    # marcas menores (sin etiqueta), saltando las que coinciden con una mayor
    tx, ty, tz = [], [], []
    for zt in _multiples(zmin, zmax, minor):
        if abs(zt - major * round(zt / major)) < 1e-6:
            continue
        tx += [x0, x0 + 0.5 * ticklen, None]; ty += [y0, y0, None]; tz += [zt, zt, None]
    traces.append(go.Scatter3d(
        x=tx, y=ty, z=tz, mode="lines",
        line=dict(color="black", width=1), hoverinfo="skip", showlegend=False))

    # marcas mayores + etiquetas
    mx, my, mz = [], [], []
    lx, ly, lz, ltext = [], [], [], []
    for zt in _multiples(zmin, zmax, major):
        mx += [x0, x0 + ticklen, None]; my += [y0, y0, None]; mz += [zt, zt, None]
        lx.append(x0 + ticklen + 40.0); ly.append(y0); lz.append(zt)
        ltext.append(str(int(round(zt))))
    traces.append(go.Scatter3d(
        x=mx, y=my, z=mz, mode="lines",
        line=dict(color="black", width=2), hoverinfo="skip", showlegend=False))
    traces.append(go.Scatter3d(
        x=lx, y=ly, z=lz, mode="text", text=ltext,
        textposition="middle right", textfont=dict(size=11, color="black"),
        hoverinfo="skip", showlegend=False))

    # etiqueta "z [m]" arriba de la regla
    traces.append(go.Scatter3d(
        x=[x0], y=[y0], z=[zmax + 0.10 * (zmax - zmin)], mode="text",
        text=["z [m]"], textposition="top center",
        textfont=dict(size=12, color="black"), hoverinfo="skip", showlegend=False))
    return traces


def _track_segment(event, default_length=1000.0):
    """Extremos (a, b) de la traza; b es la punta hacia adelante (propagacion)."""
    d = _direction_vector(event.zenith, event.azimuth)
    p0 = np.array(event.vertex, dtype=float)
    L = event.length
    if not np.isfinite(L) or L == 0:
        return p0 - default_length * d, p0 + default_length * d
    return p0, p0 + L * d


def _detector_traces(geo):
    """Lineas de strings + puntos de DOMs + jaula del contorno (todo tenue)."""
    traces = []

    # lineas verticales por string
    sx, sy, sz = [], [], []
    for x, y, zlo, zhi in geo.string_lines():
        sx += [x, x, None]; sy += [y, y, None]; sz += [zlo, zhi, None]
    traces.append(go.Scatter3d(
        x=sx, y=sy, z=sz, mode="lines",
        line=dict(color="rgba(105,105,105,0.65)", width=2),
        hoverinfo="skip", showlegend=False))

    # cada DOM como un punto pequeño
    traces.append(go.Scatter3d(
        x=geo.x, y=geo.y, z=geo.z, mode="markers",
        marker=dict(size=1.2, color="rgba(110,110,110,0.6)"),
        hoverinfo="skip", showlegend=False))

    # jaula: anillos superior e inferior del casco convexo + aristas verticales
    hull = geo.hull_xy()
    zmin, zmax = float(geo.z.min()), float(geo.z.max())
    for zc in (zmin, zmax):
        traces.append(go.Scatter3d(
            x=hull[:, 0], y=hull[:, 1], z=np.full(len(hull), zc), mode="lines",
            line=dict(color="rgba(90,90,90,0.75)", width=3),
            hoverinfo="skip", showlegend=False))
    ex, ey, ez = [], [], []
    for x, y in hull[:-1]:
        ex += [x, x, None]; ey += [y, y, None]; ez += [zmin, zmax, None]
    traces.append(go.Scatter3d(
        x=ex, y=ey, z=ez, mode="lines",
        line=dict(color="rgba(130,130,130,0.25)", width=1),
        hoverinfo="skip", showlegend=False))
    return traces


def _bubble_trace(event, chargescale, size_range, colorscale):
    """Esferas 3D sombreadas: radio [m] ~ sqrt(carga), color = tiempo [us].

    Se construye UNA sola malla que junta una esfera pequena por cada DOM golpeado,
    para que Plotly la ilumine (sombreado 3D real) en un solo trazo.
    """
    p = event.pulses
    t_us = (p.time - p.time.min()) / 1000.0   # us desde el primer DOM golpeado
    radius = np.clip(chargescale * np.sqrt(np.maximum(p.charge, 0.0)),
                     size_range[0], size_range[1])

    V, F = _SPHERE_V, _SPHERE_F
    K = len(V)
    centers = np.column_stack([p.x, p.y, p.z])                 # (n, 3)
    verts = (V[None, :, :] * radius[:, None, None]             # (n, K, 3)
             + centers[:, None, :]).reshape(-1, 3)
    intensity = np.repeat(t_us, K)                             # color por DOM
    offsets = (np.arange(len(radius)) * K)[:, None, None]      # (n,1,1)
    faces = (F[None, :, :] + offsets).reshape(-1, 3)

    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=intensity, colorscale=colorscale, reversescale=True,
        showscale=True, flatshading=False,
        colorbar=dict(title=dict(text="tiempo [us]", side="bottom"),
                      orientation="h", thickness=14, len=0.6,
                      x=0.5, xanchor="center", y=0.02, yanchor="bottom"),
        lighting=dict(ambient=0.5, diffuse=0.9, specular=0.4, roughness=0.4,
                      fresnel=0.2),
        lightposition=dict(x=400, y=-2000, z=1600),
        name="DOMs golpeados", hoverinfo="skip", showlegend=False)


def _track_traces(event, default_length, color, conesize):
    """La traza reconstruida: linea + punta de flecha (cono) hacia adelante."""
    a, b = _track_segment(event, default_length)
    line = go.Scatter3d(
        x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]], mode="lines",
        line=dict(color=color, width=4), name="reconstruccion")
    d = _direction_vector(event.zenith, event.azimuth)
    cone = go.Cone(
        x=[b[0]], y=[b[1]], z=[b[2]], u=[d[0]], v=[d[1]], w=[d[2]],
        sizemode="absolute", sizeref=conesize, anchor="tip",
        showscale=False, colorscale=[[0, color], [1, color]],
        hoverinfo="skip", showlegend=False)
    return [line, cone]


def event_display(event, show_track=True, chargescale=6.0, size_range=(5.0, 28.0),
                  colorscale="Rainbow", default_length=1000.0, conesize=120.0,
                  track_color="#111111", height=680):
    """Dibujar un evento. Devuelve una figura de Plotly (se muestra sola en Colab)."""
    geo = event.geometry
    traces = _detector_traces(geo)
    traces += _zaxis_traces(geo)
    traces.append(_bubble_trace(event, chargescale, size_range, colorscale))
    if show_track and np.isfinite(event.zenith):
        traces += _track_traces(event, default_length, track_color, conesize)

    title = f"Run {event.run} - evento {event.event}"
    if event.subevent:
        title += f".{event.subevent}"
    title += f"   (cenit {event.zenith_deg:.0f}deg, rlogl {event.rlogl:.2f})"

    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=15)),
        height=height, margin=dict(l=0, r=0, t=40, b=50),
        showlegend=True,
        legend=dict(x=0.98, xanchor="right", y=0.98, yanchor="top",
                    bgcolor="rgba(255,255,255,0.7)"),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False),   # regla de z propia, fija a la derecha
            aspectmode="data",
            camera=dict(eye=dict(x=0.25, y=-1.7, z=0.6),
                        up=dict(x=0, y=0, z=1))))
    return fig
