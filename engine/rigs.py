"""Skeletons for the cut-out characters.

Every bone is a soft capsule (pivot -> tip, radius `r`) painted on the flat picture; the picture is split into
weighted layers (one per bone + the untouched body), each layer is warped by its own matrix and the layers are
summed (see Sprite.apply_bones).  Coordinates are normalised to the character's body box:
u = 0..1 left..right, v = 0..1 top..bottom; radii are fractions of the body HEIGHT.

Bone fields
    n       name
    p, t    pivot (joint) and tip
    r       capsule radius, soft = extra falloff (fraction of r)
    parent  optional parent bone (its motion carries this bone)
    ch      motion channel that drives it ("stride", "legA", "wave", "jaw", ...), value roughly -1..1 (or 0..1)
    rot0    rest-offset angle in degrees (+ is clockwise on screen); rot degrees per unit of channel
    tx, ty  translation per unit of channel in body-height fractions;  rect = "pos" / "neg" half-wave rectifies
    idle    (degrees, hertz): gentle always-on oscillation so the limb is never frozen
"""

# padding (fraction of the largest sprite side) added around rigged sprites so limbs can swing outside the picture
PAD = {
    "nick_walk": 0.30, "lion_full": 0.22, "turtle": 0.18, "flash": 0.20, "clawhauser": 0.20, "beaver": 0.22,
    "judy_full": 0.40, "nick_full": 0.15,
}

BONES = {
    # Nick mid-stride, faces LEFT.  "stride" 0 = the pose as drawn, 1 = the mirrored stride
    "nick_walk": [
        dict(n="legF", p=(0.56, 0.75), t=(0.38, 0.965), r=0.075, vmin=0.74, umax=0.60, ch="stride", rot=-32),
        dict(n="legB", p=(0.66, 0.75), t=(0.93, 0.93), r=0.075, vmin=0.74, umin=0.60, ch="stride", rot=34),
        dict(n="armF", p=(0.33, 0.53), t=(0.07, 0.69), r=0.05, soft=0.3, ch="stride", rot=-30),
    ],
    # Lion walking toward the camera: knees lift alternately, fists swing
    "lion_full": [
        dict(n="legA", p=(0.40, 0.84), t=(0.37, 0.985), r=0.07, vmin=0.80, umax=0.48, ch="legA", ty=-0.04, rect="pos"),
        dict(n="legB", p=(0.54, 0.84), t=(0.57, 0.985), r=0.07, vmin=0.80, umin=0.48, ch="legB", ty=-0.04, rect="pos"),
        dict(n="armA", p=(0.21, 0.42), t=(0.14, 0.65), r=0.115, soft=0.9, ch="armA", rot=9, idle=(2, 0.4)),
        dict(n="armB", p=(0.78, 0.43), t=(0.80, 0.66), r=0.115, soft=0.9, ch="armB", rot=-9, idle=(2, 0.45)),
        dict(n="jaw", mode="warp", p=(0.47, 0.262), t=(0.47, 0.305), r=0.05, soft=1.3, ch="jaw", ty=0.030),
    ],
    "turtle": [
        dict(n="head", p=(0.335, 0.40), t=(0.12, 0.20), r=0.25, umax=0.335, soft=0.25, ch="head", rot=9, idle=(3, 0.5)),
        dict(n="legA", p=(0.40, 0.60), t=(0.24, 0.93), r=0.12, vmin=0.56, umax=0.47, ch="legA", rot=24, idle=(3, 0.6)),
        dict(n="legB", p=(0.62, 0.74), t=(0.72, 0.96), r=0.09, vmin=0.68, umin=0.55, umax=0.80, ch="legB", rot=22, idle=(3, 0.6)),
        dict(n="legC", p=(0.88, 0.70), t=(0.96, 0.90), r=0.10, vmin=0.66, umin=0.82, ch="legC", rot=22, idle=(3, 0.6)),
    ],
    # Flash: the raised hand waves, slowly
    "flash": [
        dict(n="wave", p=(0.78, 0.57), t=(0.83, 0.08), r=0.14, umin=0.715, ch="wave", rot=16, idle=(9, 0.33)),
        dict(n="mug", p=(0.22, 0.58), t=(0.08, 0.80), r=0.10, ch="armA", rot=6, idle=(3, 0.25)),
    ],
    "clawhauser": [
        dict(n="wave", p=(0.30, 0.40), t=(0.14, 0.19), r=0.12, umax=0.235, ch="wave", rot=18, idle=(11, 0.55)),
        dict(n="box", p=(0.62, 0.52), t=(0.70, 0.68), r=0.12, ch="armA", rot=7, idle=(3, 0.3)),
    ],
    "beaver": [
        dict(n="wave", p=(0.38, 0.30), t=(0.05, 0.21), r=0.095, ch="wave", rot=18, idle=(9, 0.65)),
    ],
    # Judy: the free arm swings, legs open into a star-jump in the air ("spread")
    "judy_full": [
        dict(n="armA", p=(0.22, 0.54), t=(0.10, 0.80), r=0.06, umax=0.26, vmin=0.5, ch="armA", rot=16, idle=(4, 0.3)),
        dict(n="legA", p=(0.40, 0.70), t=(0.26, 0.99), r=0.08, soft=0.2, vmin=0.69, umax=0.46, ch="spread", rot=34),
        dict(n="legB", p=(0.55, 0.70), t=(0.80, 0.99), r=0.08, soft=0.2, vmin=0.69, umin=0.46, ch="spread", rot=-34),
    ],
}
