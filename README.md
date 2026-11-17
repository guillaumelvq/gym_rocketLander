# Rocket RL Lab

Faire atterrir une fusée sur un bateau par apprentissage par renforcement (RL), et surtout
**comprendre comment la reward façonne le comportement** : atterrir au centre, s'arrêter au
bord, faire un salto, se poser sur un pied puis l'autre… ou exploiter une reward mal écrite
(*reward hacking*).

Le projet contient :

- **l'environnement** `RocketLander` (physique Box2D), configurable : taille de la zone,
  point de départ, vitesse, niveaux de difficulté…
- **une application web locale** pour piloter soi-même, comparer des agents côte à côte,
  écrire une reward et lancer des entraînements avec des courbes en direct ;
- **un notebook** pour l'atelier, et des scripts en ligne de commande.

Basé sur le [gym_rocketLander d'EmbersArc](https://github.com/EmbersArc/gym_rocketLander).

---

## Installation

Il faut seulement [uv](https://docs.astral.sh/uv/). Aucun droit administrateur n'est
nécessaire, et tout tourne sur CPU.

```bash
git clone https://github.com/guillaumelvq/gym_rocketLander.git
cd gym_rocketLander
uv sync
```

`uv sync` crée `.venv/` (Python 3.12) avec les versions exactes de `uv.lock` : torch (CPU),
stable-baselines3, gymnasium, FastAPI, Jupyter…

> Ne mets pas le dépôt dans un dossier OneDrive : la synchronisation du `.venv` (plusieurs Go)
> ralentit tout et provoque des fichiers verrouillés.

## Lancer l'application

```bash
uv run python -m rocketlander.app
```

Le navigateur s'ouvre sur <http://127.0.0.1:8000>. Pour changer de port, passe
`--port 8080` ; pour ne pas ouvrir le navigateur, `--no-browser`.

L'application n'écoute que sur la machine locale, parce qu'elle exécute le code de reward
tapé dans la page.

Pour une démo, une URL comme `http://127.0.0.1:8000/?autoplay&speed=2&camera=follow` lance
directement les arènes.

### L'écran

| Zone | Rôle |
|---|---|
| **Défi** (à gauche) | Choisit un exercice. Il règle le scénario et charge une reward de départ. |
| **Scénario** | Niveau de difficulté et réglages fins : largeur de la zone, position, hauteur et vitesse de départ, vitesse maximale au contact, longueur des jambes, fin à l'atterrissage, carburant… Un point orange signale une valeur modifiée par rapport au niveau. |
| **Reward** | Le code Python de ta reward, et les poids de la reward de base. |
| **Onglet Arène** | Une ou plusieurs simulations en direct, chacune avec son pilote. |
| **Onglet Entraînements** | Lancer un entraînement et suivre les courbes d'apprentissage. |

### Arène : regarder, comparer, piloter

- Chaque arène a son **pilote** :
  - **Humain** : toi, au clavier ;
  - **Autopilote** : un contrôleur programmé à la main, sans IA, utile comme référence ;
  - **Aléatoire** : des actions au hasard ;
  - les **agents entraînés**.
- **▶ Lancer** (ou `Espace`) démarre toutes les arènes avec la même **graine**, donc les
  mêmes conditions de départ : la comparaison est équitable.
- **+ Arène** en ajoute une. Par exemple, l'agent « centre » contre l'agent « carburant ».
- **Vitesse** va de ×0.25 à ×8. **Caméra** : *Auto* zoome près du sol, *Suivi* reste zoomée
  sur la fusée.
- **Enchaîner** relance un épisode dès que le précédent se termine, avec une nouvelle graine.
  Le compteur « réussites » mesure la fiabilité.
- Un agent entraîné joue dans **le scénario de son entraînement**, pas celui de la barre
  latérale.

Pilotage clavier : `↑` / `↓` pour la poussée, `←` / `→` pour orienter le moteur, `Q` / `D`
pour les propulseurs latéraux. Clique d'abord dans l'arène du pilote humain.

Le HUD affiche l'altitude, les vitesses, l'inclinaison, le nombre de tours, la poussée, le
carburant et le contact de chaque pied. La vitesse verticale passe au rouge si elle dépasse
la limite de casse près du sol.

### Adapter la reward

La reward est une fonction Python, appelée **60 fois par seconde de simulation** :

```python
def reward(s, base_reward):
    r = base_reward                      # partir de la reward de base…
    if s["landed_full"]:                 # …et ajouter ce qu'on veut encourager
        r += 10 * (1 - abs(s["x_on_ship"]))
    return r
```

- `base_reward` est la reward par défaut. Elle guide vers le bateau en récompensant les
  progrès (se rapprocher, ralentir, rester droit), donne un gros bonus à l'atterrissage et
  pénalise le crash et le temps écoulé.
- `s` contient l'état de la fusée à cet instant. La liste complète est dans la page, sous
  « Ce que contient `s` ». Les plus utiles :

  | clé | signification |
  |---|---|
  | `altitude` | hauteur au-dessus du pont, en m |
  | `x_on_ship` | position sur le bateau : 0 au centre, ±1 aux bords |
  | `velocity_x`, `velocity_y`, `speed` | vitesses (×40 m/s), `velocity_y < 0` en chute |
  | `angle`, `angular_velocity` | inclinaison (×π, 0 = droit), vitesse de rotation |
  | `rotation` | nombre de tours depuis le départ (pour le salto) |
  | `left_leg_contact`, `right_leg_contact` | chaque pied touche-t-il le bateau ? |
  | `landed`, `landed_full` | posé ; posé depuis 1 s (réussite) |
  | `game_over`, `broken_leg`, `outside` | crash, pied cassé, sortie de l'écran |
  | `fuel`, `throttle` | carburant restant (0 à 1), poussée |
  | `step` | instant de l'épisode (60 par seconde) |
  | `memory` | dictionnaire libre pour retenir des choses pendant l'épisode |

- `s["memory"]` est vidé à chaque épisode. Il sert aux objectifs « en plusieurs temps » :
  un pied puis l'autre, un salto puis l'atterrissage… Le défi *Un pied puis l'autre* en
  montre l'usage.
- Les **poids de la reward de base** (« Poids de la reward de base », sous l'éditeur) changent
  `base_reward` sans écrire de code : bonus d'atterrissage, pénalité de crash, guidage…

**Tester sa reward avant d'entraîner.** Lance une arène en pilote *Humain* ou *Autopilote* :
le compteur **« votre reward »** affiche en direct ce que ta fonction rapporte. Essaie le
comportement que tu veux, puis un comportement que tu ne veux pas : si le second rapporte
plus, l'agent le trouvera. Une erreur dans le code (clé inconnue, faute de frappe…) s'affiche
sous l'arène.

**Bonnes pratiques**

- Des petites valeurs à chaque instant (de l'ordre de 0,01), et les gros bonus ou pénalités
  seulement sur les événements de fin (`landed_full`, `game_over`).
- Se demander : « y a-t-il un moyen de gagner des points sans faire ce que je veux ? » Si
  oui, l'agent le trouvera. C'est l'objet du défi **Démo : reward hacking** : toucher le
  bateau rapporte des points et l'épisode ne s'arrête pas à l'atterrissage. Que fait
  l'agent ?

### Entraîner

1. Choisis un **défi**, règle le **scénario**, écris la **reward**.
2. Onglet **Entraînements** : donne un nom, choisis la durée, puis **Lancer l'entraînement**.
   Le code de la reward est vérifié avant le lancement.
3. La carte du run montre la progression et la courbe du **taux d'atterrissage**, des
   **crashs** et des **temps écoulés** sur les 100 derniers épisodes. À la fin, une
   évaluation sur 100 épisodes s'affiche.
4. **Voir dans l'arène** fait jouer le meilleur modèle. **Charger sa reward** remet le code
   et le scénario de ce run dans la barre latérale.

Plusieurs entraînements peuvent tourner en parallèle, chacun sur environ un cœur CPU (jusqu'à
nombre de cœurs − 2). Les résultats sont dans `runs/<nom>/` : `meta.json`, `reward.py`,
`best_model.zip`, `last_model.zip`, `progress.json`, `eval.json` et `train.log`.

## Ligne de commande

```bash
# entraîner (mêmes options que l'interface)
uv run python scripts/train.py --level 1 --timesteps 500000 --name mon-agent \
    --config '{"start_height": 0.35, "start_speed": 10}' \
    --reward-code ma_reward.py

# évaluer sur 100 épisodes et enregistrer une vidéo
uv run python scripts/evaluate.py runs/mon-agent/best_model.zip --level 1 --video atterrissage.mp4

# courbes détaillées
uv run tensorboard --logdir runs

# tests
uv run pytest tests/test_rocket_lander.py tests/test_gymnasium_env.py
```

Le notebook de l'atelier est `notebooks/rocket_lander_workshop.ipynb`. Ouvre-le dans VS Code
et choisis le noyau `.venv` du dépôt.

En Python :

```python
from rocketlander import make_env

env = make_env(level=3, config={"ship_width": 0.6}, reward_fn=lambda s, base: base)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## L'environnement

**Actions** (mode continu, chacune entre -1 et 1) : orientation du moteur, poussée (≤ 0 coupe
le moteur), propulseurs latéraux (< -0,5 à gauche, > 0,5 à droite). L'agent décide toutes les
4 frames (*frame skip*). En mode discret, il y a 7 actions : moteur ±, poussée ±, propulseur
gauche ou droit, rien.

**Niveaux** (préréglages du scénario, chaque niveau est au moins aussi dur que le précédent) :

| Niveau | Règles |
|---|---|
| 0 | Les pieds ne cassent jamais : même un contact à pleine vitesse « atterrit » |
| 1 – 9 | Les pieds cassent au-delà d'une vitesse verticale au contact : 7,6 m/s au niveau 1, 0,4 m/s de moins par niveau (4,4 m/s au niveau 9). Départ de plus en plus aléatoire |
| 10 | Même limite de 4,4 m/s, et les jambes cassent si elles sont trop comprimées |
| 11 – 20 | Jambes de plus en plus courtes (×0,93 → ×0,3) |

Pourquoi c'est dur : la fusée tombe à 40 m/s et le moteur à fond ne la freine que d'environ
10 m/s². Il faut déclencher le freinage au bon moment, à quelques mètres près. L'autopilote
(`rocketlander/autopilot.py`) montre que c'est faisable.

**Défis fournis** (`rocketlander/challenges.py`) : Atterrir, Viser le centre, S'arrêter au
bord, Faire un salto, Un pied puis l'autre, Vol stationnaire, Économiser le carburant, Démo :
reward hacking. Ils partent d'un départ facilité (plus bas, plus lent) pour apprendre en
quelques minutes sur un portable.

## Organisation du code

```
gym/envs/box2d/rocket_lander.py   physique, scénarios (DEFAULT_CONFIG, level_config), reward de base (REWARD)
rocketlander/env.py               wrapper gymnasium : frame skip, reward personnalisée, make_env()
rocketlander/challenges.py        défis de l'atelier
rocketlander/training.py          callbacks, évaluation, vidéo
rocketlander/autopilot.py         contrôleur programmé (référence)
rocketlander/app/                 application web (FastAPI + page HTML/JS sans dépendance)
scripts/train.py, evaluate.py     ligne de commande
notebooks/                        notebook de l'atelier
```

Le dossier `gym/` est un fork de Gym 0.17, conservé pour la physique et le rendu d'origine.
Le reste du projet passe par le wrapper gymnasium (`rocketlander`).
