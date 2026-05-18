# Continual Rehearsal Engine

Prevents catastrophic forgetting in neural networks for ETF return prediction. Implements Experience Replay (store & replay past samples) or Elastic Weight Consolidation (penalise changes to important parameters). Model is updated daily, preserving historical knowledge.

- **Model:** MLP with 2 hidden layers
- **Methods:** replay (buffer) or ewc (regularisation)
- **Features:** previous day's return of the ETF + macro levels
- **Output:** top 3 ETFs per universe by predicted return

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
