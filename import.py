import bpy, bmesh, json, math, os
from mathutils import Vector

# ------------------ CONFIG ------------------
# Use forward slashes on Windows:
model_path  = "C:/Users/OneDrive/Desktop/2dgs/out5/model.json"
units_width = 2.0
sigma_clip  = 3.0
use_cycles  = False
# --------------------------------------------

def fail(msg):
    raise RuntimeError(msg)

def make_quad_with_uv(name="GaussianSprite"):
    # fully rebuild to avoid half-initialized state
    old = bpy.data.objects.get(name)
    if old:
        for coll in list(old.users_collection):
            coll.objects.unlink(old)
        if old.data and old.data.users == 1:
            bpy.data.meshes.remove(old.data)
        bpy.data.objects.remove(old)

    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()

    v0 = bm.verts.new(Vector((-0.5, -0.5, 0.0)))
    v1 = bm.verts.new(Vector(( 0.5, -0.5, 0.0)))
    v2 = bm.verts.new(Vector(( 0.5,  0.5, 0.0)))
    v3 = bm.verts.new(Vector((-0.5,  0.5, 0.0)))
    f = bm.faces.new((v0, v1, v2, v3))
    bm.normal_update()

    uv_layer = bm.loops.layers.uv.new("UVMap")
    # set UVs per-loop (order matches face creation)
    f.loops[0][uv_layer].uv = (0.0, 0.0)
    f.loops[1][uv_layer].uv = (1.0, 0.0)
    f.loops[2][uv_layer].uv = (1.0, 1.0)
    f.loops[3][uv_layer].uv = (0.0, 1.0)

    bm.to_mesh(mesh); bm.free()
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    # sanity checks
    if len(mesh.polygons) == 0 or len(mesh.loops) == 0:
        fail("Sprite quad has no polygons/loops; UVs can’t be assigned.")
    return obj

def eig2_2x2(a,b,c):
    tr = a+b
    det = a*b - c*c
    s = math.sqrt(max(tr*tr - 4.0*det, 0.0))
    l1 = 0.5*(tr + s); l2 = 0.5*(tr - s)
    if abs(c) > 1e-12:
        v1x, v1y = (l1 - b), c
    else:
        v1x, v1y = (1.0, 0.0) if a >= b else (0.0, 1.0)
    n = math.hypot(v1x, v1y); v1x /= (n or 1.0); v1y /= (n or 1.0)
    return l1, l2, v1x, v1y

# 0) Load JSON
if not os.path.isfile(model_path):
    fail(f"model.json not found: {model_path}")
with open(model_path, "r") as f:
    model = json.load(f)

W = int(model.get("w", 0)); H = int(model.get("h", 0))
comps = model.get("c", [])
if W <= 0 or H <= 0:
    fail(f"Invalid width/height in model.json (W={W}, H={H}).")
if not comps:
    fail("model.json has zero components; nothing to import.")

z_ref = float(model.get("z", 2.0))
fog   = float(model.get("fog", 0.0))
px_to_m = units_width / max(1, W)

print(f"[OK] JSON loaded. W={W} H={H} N={len(comps)}")

# 1) Build helper quad with UVs
quad = make_quad_with_uv("GaussianSprite")
print(f"[OK] Quad built. faces={len(quad.data.polygons)} loops={len(quad.data.loops)}")

# 2) Make material
mat = bpy.data.materials.get("GaussianMat") or bpy.data.materials.new("GaussianMat")
mat.use_nodes = True
ng = mat.node_tree
for n in list(ng.nodes): ng.nodes.remove(n)
nodes, links = ng.nodes, ng.links

out = nodes.new("ShaderNodeOutputMaterial"); out.location = (600, 0)
mix = nodes.new("ShaderNodeMixShader");      mix.location = (360, 0)
transp = nodes.new("ShaderNodeBsdfTransparent"); transp.location = (120, 80)
emit = nodes.new("ShaderNodeEmission");      emit.location = (120, -120)
attr_col = nodes.new("ShaderNodeAttribute"); attr_col.attribute_name = "col";         attr_col.location = (-820, -160)
attr_amp = nodes.new("ShaderNodeAttribute"); attr_amp.attribute_name = "amp";        attr_amp.location = (-820, -340)
attr_a   = nodes.new("ShaderNodeAttribute"); attr_a.attribute_name   = "alpha_final";attr_a.location   = (-820,  60)

uv  = nodes.new("ShaderNodeUVMap"); uv.location = (-820, 260)
sepxy = nodes.new("ShaderNodeSeparateXYZ"); sepxy.location = (-620, 260)
subX = nodes.new("ShaderNodeMath"); subX.operation='SUBTRACT'; subX.inputs[1].default_value=0.5; subX.location=(-460, 340)
subY = nodes.new("ShaderNodeMath"); subY.operation='SUBTRACT'; subY.inputs[1].default_value=0.5; subY.location=(-460, 200)
mulX = nodes.new("ShaderNodeMath"); mulX.operation='MULTIPLY'; mulX.inputs[1].default_value=2.0; mulX.location=(-300, 340)
mulY = nodes.new("ShaderNodeMath"); mulY.operation='MULTIPLY'; mulY.inputs[1].default_value=2.0; mulY.location=(-300, 200)
powX = nodes.new("ShaderNodeMath"); powX.operation='POWER';    powX.inputs[1].default_value=2.0; powX.location=(-140, 340)
powY = nodes.new("ShaderNodeMath"); powY.operation='POWER';    powY.inputs[1].default_value=2.0; powY.location=(-140, 200)
addR2 = nodes.new("ShaderNodeMath"); addR2.operation='ADD'; addR2.location=(20, 270)
mulHf = nodes.new("ShaderNodeMath"); mulHf.operation='MULTIPLY'; mulHf.inputs[1].default_value=-0.5; mulHf.location=(180, 270)
exp   = nodes.new("ShaderNodeMath"); exp.operation='EXPONENT'; exp.location=(340, 270)

mulCol = nodes.new("ShaderNodeVectorMath"); mulCol.operation='MULTIPLY'; mulCol.location=(-600, -260)

links.new(uv.outputs['UV'], sepxy.inputs[0])
links.new(sepxy.outputs['X'], subX.inputs[0]); links.new(sepxy.outputs['Y'], subY.inputs[0])
links.new(subX.outputs[0], mulX.inputs[0]);    links.new(subY.outputs[0], mulY.inputs[0])
links.new(mulX.outputs[0],  powX.inputs[0]);   links.new(mulY.outputs[0],  powY.inputs[0])
links.new(powX.outputs[0],  addR2.inputs[0]);  links.new(powY.outputs[0],  addR2.inputs[1])
links.new(addR2.outputs[0], mulHf.inputs[0]);  links.new(mulHf.outputs[0], exp.inputs[0])

links.new(attr_col.outputs['Color'], mulCol.inputs[0])
links.new(attr_amp.outputs['Fac'],   mulCol.inputs[1])
links.new(mulCol.outputs['Vector'],  emit.inputs['Color'])

mulAlpha = nodes.new("ShaderNodeMath"); mulAlpha.operation='MULTIPLY'; mulAlpha.location=(340, 80)
links.new(attr_a.outputs['Fac'], mulAlpha.inputs[0])
links.new(exp.outputs[0],        mulAlpha.inputs[1])

links.new(transp.outputs['BSDF'], mix.inputs[1])
links.new(emit.outputs['Emission'], mix.inputs[2])
links.new(mulAlpha.outputs[0],     mix.inputs['Fac'])
links.new(mix.outputs['Shader'],   out.inputs['Surface'])

mat.blend_method = 'BLEND'
mat.use_backface_culling = False

# Attach material to the quad (safe: append, no indexing)
quad.data.materials.clear()
quad.data.materials.append(mat)
print("[OK] Material ready.")

# 3) Build the point cloud mesh
mesh = bpy.data.meshes.new("CloudGaussiansMesh")
obj  = bpy.data.objects.new("CloudGaussians", mesh)
bpy.context.scene.collection.objects.link(obj)

verts = []
for g in comps:
    x = (g["x"] - 0.5*W) * px_to_m
    y = (0.5*H - g["y"]) * px_to_m
    z = -float(g["z"])
    verts.append((x, y, z*0.4))

mesh.from_pydata(verts, [], [])
mesh.update()
if len(mesh.vertices) == 0:
    fail("Point mesh has 0 vertices after from_pydata().")

print(f"[OK] Point mesh built. verts={len(mesh.vertices)}")

# 4) Add per-point attributes — robust in 4.5:
#    - floats via Vertex Groups (rad_a, rad_b, theta, alpha_final, amp)
#    - color via Mesh Color Attribute "col" (FLOAT_COLOR, POINT)

import math
from mathutils import Vector

mesh.validate(clean_customdata=False)
mesh.update()

# Ensure we're in OBJECT mode to write vertex groups
bpy.context.view_layer.objects.active = obj
if bpy.context.object.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')

# Remove old vertex groups if they exist
for nm in ("rad_a","rad_b","theta","alpha_final","amp"):
    vg = obj.vertex_groups.get(nm)
    if vg: obj.vertex_groups.remove(vg)

# Create new vertex groups
vg_ra = obj.vertex_groups.new(name="rad_a")
vg_rb = obj.vertex_groups.new(name="rad_b")
vg_th = obj.vertex_groups.new(name="theta")
vg_af = obj.vertex_groups.new(name="alpha_final")
vg_am = obj.vertex_groups.new(name="amp")

n = len(mesh.vertices)

# Precompute arrays
rad_a_vals = [0.0]*n
rad_b_vals = [0.0]*n
theta_vals = [0.0]*n
alpha_vals = [0.0]*n
amp_vals   = [0.0]*n
col_vals   = [0.0]*(n*3)

for i, g in enumerate(comps):
    cov = g["cov"]; a = float(cov[0][0]); b = float(cov[1][1]); c = float(cov[0][1])
    tr = a+b; det = a*b - c*c
    s = math.sqrt(max(tr*tr - 4.0*det, 0.0))
    l1 = 0.5*(tr + s); l2 = 0.5*(tr - s)

    if abs(c) > 1e-12:
        v1x, v1y = (l1 - b), c
    else:
        v1x, v1y = ((1.0,0.0) if a >= b else (0.0,1.0))
    nrm = math.hypot(v1x, v1y) or 1.0
    v1x /= nrm; v1y /= nrm

    s1_px = math.sqrt(max(l1, 1e-12))
    s2_px = math.sqrt(max(l2, 1e-12))

    zval   = max(1e-6, float(g["z"]))
    scaleZ = z_ref / zval
    rad_a  = s1_px * sigma_clip * scaleZ * px_to_m
    rad_b  = s2_px * sigma_clip * scaleZ * px_to_m
    theta  = math.atan2(v1y, v1x)

    fog_fac = math.exp(-fog * max(0.0, zval - z_ref)) if fog > 0.0 else 1.0
    amp   = float(g["amplitude"])
    a_fin = float(g["alpha"]) * amp * fog_fac
    col   = g.get("color", [1,1,1])

    rad_a_vals[i] = rad_a
    rad_b_vals[i] = rad_b
    theta_vals[i] = theta
    alpha_vals[i] = a_fin
    amp_vals[i]   = amp
    base = i*3
    col_vals[base+0] = float(col[0])
    col_vals[base+1] = float(col[1])
    col_vals[base+2] = float(col[2])

# Assign the weights (one call per vertex per group)
for i in range(n):
    vg_ra.add([i], rad_a_vals[i], 'REPLACE')
    vg_rb.add([i], rad_b_vals[i], 'REPLACE')
    vg_th.add([i], theta_vals[i], 'REPLACE')
    vg_af.add([i], alpha_vals[i], 'REPLACE')
    vg_am.add([i], amp_vals[i],   'REPLACE')

# Color attribute "col" as FLOAT_COLOR on POINT domain
col_layer = mesh.color_attributes.get("col")
if col_layer:
    mesh.color_attributes.remove(col_layer)
col_layer = mesh.color_attributes.new(name="col", domain='POINT', type='FLOAT_COLOR')

# pack RGBA (A=1.0)
col_rgba = [0.0]*(n*4)
for i in range(n):
    r = col_vals[i*3+0]; g = col_vals[i*3+1]; b = col_vals[i*3+2]
    base = i*4
    col_rgba[base+0] = r
    col_rgba[base+1] = g
    col_rgba[base+2] = b
    col_rgba[base+3] = 1.0
col_layer.data.foreach_set("color", col_rgba)

mesh.update()
print("[OK] Attributes written via vertex groups + color attribute.")


# 5) Geometry Nodes instancer (Blender 4.5 interface API)
gn = bpy.data.node_groups.get("GN_GaussianSprites")
if gn is None:
    gn = bpy.data.node_groups.new("GN_GaussianSprites", 'GeometryNodeTree')

# Clear nodes/links
gn.nodes.clear()
gn.links.clear()

# Rebuild the group interface: one Geometry input + one Geometry output
iface = gn.interface
# (optional) wipe existing interface sockets if any
try:
    for item in list(iface.items_tree):
        iface.remove(item)
except Exception:
    pass

iface.new_socket(name="Geometry", in_out='INPUT',  socket_type='NodeSocketGeometry')
iface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

# Group I/O nodes
n_in  = gn.nodes.new("NodeGroupInput");  n_in.location  = (-800, 0)
n_out = gn.nodes.new("NodeGroupOutput"); n_out.location = ( 800, 0)

# Attribute readers (float groups)
n_attr_ra = gn.nodes.new("GeometryNodeInputNamedAttribute"); n_attr_ra.location = (-600,  200); n_attr_ra.data_type = 'FLOAT'; n_attr_ra.inputs[0].default_value = "rad_a"
n_attr_rb = gn.nodes.new("GeometryNodeInputNamedAttribute"); n_attr_rb.location = (-600,   20); n_attr_rb.data_type = 'FLOAT'; n_attr_rb.inputs[0].default_value = "rad_b"
n_attr_th = gn.nodes.new("GeometryNodeInputNamedAttribute"); n_attr_th.location = (-600, -160); n_attr_th.data_type = 'FLOAT'; n_attr_th.inputs[0].default_value = "theta"

# Instance source: the quad
n_obj = gn.nodes.new("GeometryNodeObjectInfo"); n_obj.location = (-600, -360)
n_obj.transform_space = 'RELATIVE'
n_obj.inputs['As Instance'].default_value = True
n_obj.inputs['Object'].default_value = quad

# Build instancing chain
n_inst = gn.nodes.new("GeometryNodeInstanceOnPoints"); n_inst.location = (-200, 0)
n_rot  = gn.nodes.new("GeometryNodeRotateInstances");  n_rot.location  = ( 200, 0)
n_scl  = gn.nodes.new("GeometryNodeScaleInstances");   n_scl.location  = ( 400, 0)
n_mat  = gn.nodes.new("GeometryNodeSetMaterial");      n_mat.location  = ( 600, 0);  n_mat.inputs['Material'].default_value = mat

# Helpers
n_comb = gn.nodes.new("ShaderNodeCombineXYZ"); n_comb.location = (200, -200)
# Rotation = (0, 0, theta)
n_rot_comb = gn.nodes.new("ShaderNodeCombineXYZ"); n_rot_comb.location = (0, -200)
n_rot_comb.inputs['X'].default_value = 0.0
n_rot_comb.inputs['Y'].default_value = 0.0
gn.links.new(n_attr_th.outputs['Attribute'], n_rot_comb.inputs['Z'])
gn.links.new(n_rot_comb.outputs['Vector'],   n_rot.inputs['Rotation'])

# Links (use socket indices for Group I/O in 4.5)
gn.links.new(n_in.outputs[0],               n_inst.inputs['Points'])   # Group Input → Points
gn.links.new(n_obj.outputs['Geometry'],     n_inst.inputs['Instance'])

gn.links.new(n_attr_ra.outputs['Attribute'], n_comb.inputs['X'])
gn.links.new(n_attr_rb.outputs['Attribute'], n_comb.inputs['Y'])
gn.links.new(n_comb.outputs['Vector'],      n_scl.inputs['Scale'])
gn.links.new(n_inst.outputs['Instances'],   n_rot.inputs['Instances'])
gn.links.new(n_rot.outputs['Instances'],    n_scl.inputs['Instances'])
gn.links.new(n_scl.outputs['Instances'],    n_mat.inputs['Geometry'])
gn.links.new(n_mat.outputs['Geometry'],     n_out.inputs[0])           # → Group Output

# Attach the GN group to the object’s modifier
mod = obj.modifiers.get("GaussiansGN") or obj.modifiers.new(name="GaussiansGN", type='NODES')
mod.node_group = gn

# Make sure the instancer has the material
quad.data.materials.clear()
quad.data.materials.append(mat)


print("[OK] Geometry Nodes built and assigned.")
print("SUMMARY:",
      "quad faces=", len(quad.data.polygons),
      "loops=", len(quad.data.loops),
      "cloud verts=", len(obj.data.vertices))

# Optional: engine

print("✔ Done.")


import bpy
obj = bpy.data.objects["CloudGaussians"]
mod = obj.modifiers["GaussiansGN"]
gn  = mod.node_group
ns  = gn.nodes
ln  = gn.links

# Grab existing nodes we made earlier
n_attr_ra = ns["Input Named Attribute"]      if "Input Named Attribute" in ns else None
# safer lookup by label is messy; instead find by type+name we set:
def find_by_name(prefix):
    for n in ns:
        if n.name.startswith(prefix):
            return ns[n.name]
    return None

# We know these names from the script creation:
n_attr_ra = [n for n in ns if n.bl_idname=="GeometryNodeInputNamedAttribute" and n.inputs[0].default_value=="rad_a"][0]
n_attr_rb = [n for n in ns if n.bl_idname=="GeometryNodeInputNamedAttribute" and n.inputs[0].default_value=="rad_b"][0]
n_attr_th = [n for n in ns if n.bl_idname=="GeometryNodeInputNamedAttribute" and n.inputs[0].default_value=="theta"][0]
n_inst    = [n for n in ns if n.bl_idname=="GeometryNodeInstanceOnPoints"][0]
n_rot     = [n for n in ns if n.bl_idname=="GeometryNodeRotateInstances"][0]
n_scl     = [n for n in ns if n.bl_idname=="GeometryNodeScaleInstances"][0]
n_mat     = [n for n in ns if n.bl_idname=="GeometryNodeSetMaterial"][0]

# 1) Add readers for the other attrs we need in the shader
n_attr_amp = ns.new("GeometryNodeInputNamedAttribute"); n_attr_amp.location = (-600, -340)
n_attr_amp.data_type = 'FLOAT'; n_attr_amp.inputs[0].default_value = "amp"
n_attr_af  = ns.new("GeometryNodeInputNamedAttribute"); n_attr_af.location  = (-600, -520)
n_attr_af.data_type  = 'FLOAT'; n_attr_af.inputs[0].default_value  = "alpha_final"
n_attr_col = ns.new("GeometryNodeInputNamedAttribute"); n_attr_col.location = (-600, -700)
n_attr_col.data_type = 'FLOAT_VECTOR'; n_attr_col.inputs[0].default_value = "col"

# 2) Store them on the INSTANCE domain so they travel with each quad
n_store_amp = ns.new("GeometryNodeStoreNamedAttribute"); n_store_amp.location = (600, -300)
n_store_amp.inputs["Name"].default_value = "amp"; n_store_amp.data_type='FLOAT'; n_store_amp.domain='INSTANCE'

n_store_af  = ns.new("GeometryNodeStoreNamedAttribute"); n_store_af.location  = (600, -500)
n_store_af.inputs["Name"].default_value  = "alpha_final"; n_store_af.data_type='FLOAT'; n_store_af.domain='INSTANCE'

n_store_col = ns.new("GeometryNodeStoreNamedAttribute"); n_store_col.location = (600, -700)
n_store_col.inputs["Name"].default_value = "col"; n_store_col.data_type='FLOAT_VECTOR'; n_store_col.domain='INSTANCE'

# 3) Realize instances so the material can read the stored attributes
n_realize = ns.new("GeometryNodeRealizeInstances"); n_realize.location = (820, -40)

# 4) Rewire chain: n_scl -> stores -> realize -> set material
# disconnect old link into Set Material
for l in list(n_mat.inputs['Geometry'].links):
    ln.remove(l)

ln.new(n_scl.outputs['Instances'], n_store_amp.inputs['Geometry'])
ln.new(n_attr_amp.outputs['Attribute'], n_store_amp.inputs['Value'])

ln.new(n_store_amp.outputs['Geometry'], n_store_af.inputs['Geometry'])
ln.new(n_attr_af.outputs['Attribute'],  n_store_af.inputs['Value'])

ln.new(n_store_af.outputs['Geometry'],  n_store_col.inputs['Geometry'])
ln.new(n_attr_col.outputs['Attribute'], n_store_col.inputs['Value'])

ln.new(n_store_col.outputs['Geometry'], n_realize.inputs['Geometry'])
ln.new(n_realize.outputs['Geometry'],   n_mat.inputs['Geometry'])

print("GN patched: attributes stored on instances and realized.")



