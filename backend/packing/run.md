# Setup & Run Instructions
 
## 1. Clone the Repository
 
```bash
git clone git@github.com:jesuisunananas/PackingApp.git
cd backend/packing
```
 
## 2. Create Conda Environment & Install Dependencies

```bash
conda create -n packing python=3.10
conda activate packing
pip install -r requirements.txt
```

## 3. Run Main Scripts
 
Deploy the modal workflows (detached):
 
```bash
uv run modal run --detach modal_generate_data.py
uv run modal run --detach modal_train_bc.py
uv run modal run --detach modal_train_online_sac.py
```

## 4. Run Ablation
 
```bash
python ablation.py
```