# Brief Claude Code — Pipeline Computer Vision Gemini 3.5 Flash

## Contexte

Je pars d'un notebook Colab qui fait du **zero-shot object detection** avec Google Gemini 3.5 Flash et la lib `supervision` (Roboflow). Le notebook fonctionne mais n'est pas réutilisable : tout est dans des cellules, le client est instancié via `userdata` (Colab), et les exemples sont dupliqués (~7 fois le même bloc de code).

**Objectif** : transformer ça en un repo GitHub propre, packagé, testable, avec une CLI et une API Python réutilisable.

## Stack cible

- **Python 3.11+**
- **uv** pour la gestion des dépendances (ou `pip` + `pyproject.toml` PEP 621 si tu préfères, mais `uv` préféré)
- **google-genai** (SDK officiel Gemini)
- **supervision** (branch `add-gemini-3.5-vlm-support` pour le moment — note-le dans le README)
- **pydantic v2** + **pydantic-settings** pour la config
- **typer** ou **click** pour la CLI (préfère `typer`)
- **pytest** pour les tests
- **ruff** + **black** pour lint/format
- **python-dotenv** (chargé par pydantic-settings)

## Structure cible

```
gemini-vision-pipeline/
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
├── .python-version
├── src/
│   └── gemini_vision/
│       ├── __init__.py
│       ├── config.py              # Settings via pydantic-settings
│       ├── client.py              # Wrapper du client Gemini
│       ├── prompts.py             # Templates de prompt
│       ├── schemas.py             # Modèles Pydantic (Detection, etc.)
│       ├── detector.py            # Classe ObjectDetector (cœur du pipeline)
│       ├── annotator.py           # Visualisation via supervision
│       └── cli.py                 # Point d'entrée CLI (typer)
├── examples/
│   ├── 01_single_class.py         # Cas mono-classe (ballons, oiseaux…)
│   ├── 02_multi_class.py          # Cas multi-classes (avocats, voitures par voie)
│   └── 03_structured_output.py    # Structured output pour scènes denses
├── notebooks/
│   └── original_exploration.ipynb # le notebook d'origine, gardé pour référence
├── scripts/
│   └── download_examples.sh       # télécharge les images d'exemple
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_prompts.py
│   ├── test_schemas.py
│   └── test_detector.py           # mocker l'API Gemini
└── assets/                        # images d'exemple (gitignore les .jpg si trop lourds)
```

## Spécifications détaillées par module

### `src/gemini_vision/config.py`

Settings via `pydantic-settings`. Lit `.env` automatiquement.

```python
class Settings(BaseSettings):
    google_api_key: SecretStr
    gemini_model: str = "gemini-3.5-flash"
    default_temperature: float = 0.0
    default_thinking_budget: int = 0  # 0 = thinking off (rapide)

    model_config = SettingsConfigDict(env_file=".env", env_prefix="")
```

Fonction `get_settings()` cachée avec `@lru_cache`.

### `src/gemini_vision/schemas.py`

Garde le modèle `Detection` du notebook. Ajoute aussi un `DetectionResult` qui encapsule la liste + métadonnées (modèle utilisé, temps de réponse, classes recherchées).

```python
class Detection(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    box_2d: list[int] = Field(min_length=4, max_length=4)

class DetectionResult(BaseModel):
    detections: list[Detection]
    classes: list[str]
    model: str
    image_size: tuple[int, int]
    raw_response: str | None = None
```

### `src/gemini_vision/prompts.py`

Migre `DETECTION_PROMPT_TEMPLATE` et `build_detection_prompt` tels quels. Garde le template en constante module-level. Type-hint propre.

### `src/gemini_vision/client.py`

Singleton (via `@lru_cache`) qui retourne un client `genai.Client` configuré avec la clé API depuis `Settings`. **Ne jamais** instancier le client depuis Colab `userdata` — c'est la dette principale du notebook.

### `src/gemini_vision/detector.py`

Cœur du pipeline. Classe `ObjectDetector` avec :

```python
class ObjectDetector:
    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        thinking_budget: int = 0,
    ): ...

    def detect(
        self,
        image: Image.Image | str | Path,
        classes: list[str],
        structured_output: bool = False,
    ) -> DetectionResult: ...

    def detect_and_annotate(
        self,
        image: Image.Image | str | Path,
        classes: list[str],
        with_labels: bool = True,
        structured_output: bool = False,
    ) -> tuple[DetectionResult, Image.Image]: ...
```

Accepte `Image`, `str` ou `Path` pour l'image. Si `structured_output=True`, utilise `response_mime_type="application/json"` + `response_schema=list[Detection]` (cas dense scenes, voir exemple "Person" du notebook).

Convertit le résultat via `sv.Detections.from_vlm(vlm=sv.VLM.GOOGLE_GEMINI_3_5, ...)`.

### `src/gemini_vision/annotator.py`

Migre `COLOR` (la `ColorPalette`) et `annotate_image`. Garde l'API simple : `annotate(image, detections, with_labels=True) -> Image.Image`. Ne pas downsizer à 1000x1000 par défaut — rendre ça optionnel via paramètre `max_size`.

### `src/gemini_vision/cli.py`

CLI `typer` avec une commande `detect` :

```bash
gemini-vision detect path/to/image.jpg --classes "car,truck,bus" --output annotated.jpg
gemini-vision detect image.jpg --classes "person" --structured  # forced JSON
gemini-vision detect image.jpg --classes "car" --no-labels --json-out detections.json
```

Options principales :
- `--classes` : liste séparée par virgules
- `--output` / `-o` : chemin de sortie pour l'image annotée
- `--json-out` : exporter les détections en JSON
- `--structured` : flag pour forcer JSON output
- `--no-labels` : pas de labels sur les bboxes
- `--model` : override du modèle

Entrypoint à déclarer dans `pyproject.toml` : `gemini-vision = "gemini_vision.cli:app"`.

### `examples/*.py`

Trois scripts autonomes qui montrent les 3 cas d'usage du notebook, **sans duplication** : ils importent depuis `gemini_vision` et appellent l'API publique. ~15 lignes chacun.

### `tests/`

- `test_prompts.py` : vérifier que `build_detection_prompt` injecte bien les classes
- `test_schemas.py` : valider les contraintes Pydantic (confidence dans [0,1], box_2d de longueur 4)
- `test_detector.py` : mocker `client.models.generate_content` avec `monkeypatch` ou `unittest.mock`, vérifier que le pipeline parse correctement une réponse JSON et appelle `sv.Detections.from_vlm`

Ne **pas** faire d'appels API réels dans les tests.

### `pyproject.toml`

- Métadonnées projet (name, version, description, authors)
- Dépendances principales
- Dépendances dev dans `[project.optional-dependencies]` ou `[dependency-groups]` (uv)
- Configuration `[tool.ruff]`, `[tool.pytest.ini_options]`
- Entrypoint CLI

Pour `supervision`, utiliser la dépendance Git tant que la branche n'est pas mergée :
```toml
supervision = { git = "https://github.com/roboflow/supervision.git", branch = "add-gemini-3.5-vlm-support" }
```
Note dans le README : « remplacer par la version PyPI dès que le support Gemini 3.5 est mergé ».

### `.env.example`

```
GOOGLE_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.5-flash
```

### `.gitignore`

Standard Python + `.env`, `*.jpg` dans `assets/` si tu veux pas committer les images, `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`.

### `scripts/download_examples.sh`

Reprends les `wget` du notebook. Met les fichiers dans `assets/`.

### `README.md`

Sections :
1. **Description** courte (1-2 phrases)
2. **Features** : zero-shot detection, structured output, CLI, API Python, support de N classes
3. **Installation** : `uv sync` ou `pip install -e .`
4. **Configuration** : copier `.env.example` → `.env`, ajouter sa clé Gemini (lien vers AI Studio)
5. **Usage CLI** avec 2-3 exemples
6. **Usage Python** avec un snippet `ObjectDetector(...).detect_and_annotate(...)`
7. **Examples** : pointer vers `examples/`
8. **Project structure** (arbre simplifié)
9. **Roadmap** / Notes (mention de la branche supervision)
10. **License** (MIT par défaut, à confirmer)

## Plan d'exécution suggéré

Procède dans cet ordre, en commitant à chaque étape :

1. **Bootstrap** : `pyproject.toml`, `.gitignore`, `.env.example`, structure de dossiers vides
2. **Schemas + prompts + config** : les modules sans dépendance, faciles à tester
3. **Tests unitaires** sur prompts et schemas (TDD-friendly)
4. **Client + detector + annotator** : le cœur métier
5. **Test du detector** avec mock
6. **CLI** avec typer
7. **Examples** : les 3 scripts
8. **Notebook** : déplacer l'original dans `notebooks/`
9. **Script de download** + **README** complet
10. **Lint pass** : `ruff check --fix` + `ruff format`, vérifier que `pytest` passe

## Critères de qualité

- **Type hints partout**, validés avec ruff
- **Docstrings** style Google sur toutes les classes/fonctions publiques
- **Pas de duplication** : le code des 7 exemples du notebook doit tenir en 1 appel `detector.detect_and_annotate(...)`
- **Pas de secrets** dans le code ni les commits
- **CLI utilisable** sans toucher au code (`gemini-vision detect ...` marche après `uv sync`)
- **Tests verts** sans appel API réel
- **README** assez clair pour qu'un dev externe puisse cloner et lancer en 5 min

## Code de référence (depuis le notebook d'origine)

Voici le prompt template et les utilitaires à migrer **tels quels** :

```python
DETECTION_PROMPT_TEMPLATE = """
Carefully examine this image and detect ALL visible objects, including small, distant, or partially visible ones.
IMPORTANT: Focus on finding as many objects as possible, even if you are only moderately confident.
Make sure each bounding box is as tight as possible.
Valid object classes: {class_list}
For each detected object, provide:
- "label": the exact class name from the list above
- "confidence": your certainty (between 0.0 and 1.0)
- "box_2d": the bounding box [ymin, xmin, ymax, xmax] normalized to 0-1000
Detect everything that matches the valid classes. Do not be conservative; include objects even with moderate confidence.
Return a JSON array, for example:
[
    {{"label": "{class_example}", "confidence": 0.95, "box_2d": [100, 200, 300, 400]}}
]
"""

COLOR = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff66ff", "#3399ff", "#ff66b2", "#ff8080",
    "#b266ff", "#9999ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])

class Detection(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    box_2d: list[int] = Field(min_length=4, max_length=4)
```

Appel Gemini standard (mode JSON libre) :
```python
response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=[image, prompt],
    config=types.GenerateContentConfig(
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    ),
)
```

Appel Gemini en structured output (pour scènes denses) :
```python
config=types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=list[Detection],
    temperature=0,
    thinking_config=types.ThinkingConfig(thinking_budget=0)
)
```

Parsing supervision :
```python
detections = sv.Detections.from_vlm(
    vlm=sv.VLM.GOOGLE_GEMINI_3_5,
    result=response.text,
    resolution_wh=image.size,
    classes=CLASSES,
)
```

## Commande de démarrage

Quand tu attaques le projet dans Claude Code, démarre par :

```
Lis BUILD_INSTRUCTIONS.md, propose-moi l'arborescence finale et le contenu de pyproject.toml,
puis attends ma validation avant de générer le reste.
```