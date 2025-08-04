# RAG

## ⚡ Quickstart

### 1. Clone the repository

Clone the repo and enter the project directory:

```bash
git clone git@github.com:ducnd58233/rag.git
```

### 2. Set up the environment

#### Using Conda & Poetry (recommended)

```bash
conda env create -f environment.yml
conda activate rag
poetry install
```

#### Export conda env

Export your current conda environment for reproducibility:

```bash
conda env export --from-history --no-builds > environment.yml
```

#### Create .env file

Copy the example environment file:

```bash
cp .env.example .env
```

### 3. Run project

```bash
poetry run streamlit run src/main.py
```
