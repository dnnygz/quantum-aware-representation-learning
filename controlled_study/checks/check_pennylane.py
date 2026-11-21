import numpy as np, pennylane as qml
from vqc import *
dev = qml.device("default.qubit", wires=4)
L=2
@qml.qnode(dev, diff_method="parameter-shift")
def circ(x, P):
    for q in range(4): qml.RY(x[q], wires=q)
    for l in range(L):
        for q in range(4):
            qml.RY(P[l,q,0], wires=q); qml.RZ(P[l,q,1], wires=q)
        qml.CNOT(wires=[0,1]); qml.CNOT(wires=[1,2]); qml.CNOT(wires=[2,3])
    return [qml.expval(qml.PauliZ(q)) for q in range(4)]
rng=np.random.RandomState(1)
X=rng.uniform(-np.pi,np.pi,(5,4)); P=rng.uniform(-np.pi,np.pi,(L,4,2))
mine=expz(P,encode(X),L)
ref=np.array([np.array(circ(x,P)) for x in X])
print("max |expz diff|:", np.abs(mine-ref).max())
# gradient check vs finite differences of PennyLane forward
g=expz_grad(P,encode(X),L); h=1e-5; flat=P.reshape(-1); worst=0
for k in range(flat.size):
    e=np.zeros_like(flat); e[k]=h
    f=lambda PP: np.array([np.array(circ(x,PP.reshape(P.shape))) for x in X])
    fd=(f(flat+e)-f(flat-e))/(2*h)
    worst=max(worst,np.abs(fd-g[:,:,k]).max())
print("max |grad diff| vs finite diff of PennyLane:", worst)
# no-entanglement check
Z2=expz(P,encode(X),L,entangle=False)
@qml.qnode(dev)
def circ2(x,P):
    for q in range(4): qml.RY(x[q], wires=q)
    for l in range(L):
        for q in range(4):
            qml.RY(P[l,q,0], wires=q); qml.RZ(P[l,q,1], wires=q)
    return [qml.expval(qml.PauliZ(q)) for q in range(4)]
print("no-ent diff:", np.abs(Z2-np.array([np.array(circ2(x,P)) for x in X])).max())
import pennylane; print("pennylane", pennylane.__version__)
