import mujoco_py
from mujoco_py import MjViewer

# Carica il modello dal file XML
model = mujoco_py.load_model_from_path("/home/francesco/Polito/MLDL/rl_mldl_25/env/assets/hopper.xml")
#/home/francesco/Polito/MLDL/rl_mldl_25/env/assets/hopper.xml

# Crea un simulatore
sim = mujoco_py.MjSim(model)

# Crea un visualizzatore
viewer = MjViewer(sim)

for _ in range(3000):
    sim.step()
    viewer.render()

print("Posizione:", sim.data.qpos)
print("Velocità:", sim.data.qvel)

"""
Una volta che hai il modello caricato in MuJoCo, puoi farci un sacco di cose:

1. Simulare
Usi sim.step() per avanzare la simulazione di un timestep. Questo aggiorna tutte le posizioni, velocità, forze, ecc.

2. Leggere lo stato del mondo
- Posizioni (sim.data.qpos)
- Velocità (sim.data.qvel)
- Forze, contatti, sensori, tempo, ecc.

3. Applicare comandi / controlli
Se il tuo modello ha attuatori (motori, giunti...), puoi controllarli direttamente:
- sim.data.ctrl[:] = [valore1, valore2, ...]  # comandi motori
Puoi scrivere un controller PID, un policy di RL, un controller neurale, ecc.

4. Visualizzare
Con MjViewer (mujoco-py) o mujoco.viewer.launch() (per i nuovi binding).

5. Interagire con la fisica
Puoi:
- Aggiungere forze manualmente (apply_force, xfrc_applied)
- Modificare proprietà runtime (tipo massa, frizione…)
- Aggiungere contatti o constraints

6. Logging e raccolta dati
Per esperimenti o reinforcement learning:
- Salvi qpos, qvel, reward, osservazioni, ecc.
- Puoi esportare a NumPy o Pandas

7. Reset del simulatore
Utile per fare tanti episodi (tipico in RL).

8. Manipolazione avanzata
- Muovi un oggetto direttamente con mj_set_state o qpos[:] = nuovo_valore
- Calcoli cinematiche inverse / dirette
- Usi mj_forward, mj_inverse, ecc. per avanzare la simulazione manualmente
"""