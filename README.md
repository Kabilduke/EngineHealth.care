# Machine Learning model to predict the condition of the car engine.

Machine learning model was trained on a 20,000 parameter dataset to predict the health of car engines using various algorithm like KNN, SVM, XBoost Classifer and Random Forest

Model Deployment: https://huggingface.co/spaces/Kabil007/EngineHealth.care

![Model ShowCase:](https://github.com/Kabilduke/EngineHealth.care/blob/main/Output.png)

### Requirements
- scikit-learn
- streamlit

### Installation
1. Clone the repository
   
  git clone:: https://github.com/Kabilduke/EngineHealth.care.git
  cd Engine

2. Create a virtual environment and activate it:
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required packages:
   pip install requirements.txt

4. Run the streamlit app:
   streamlit run app.py
