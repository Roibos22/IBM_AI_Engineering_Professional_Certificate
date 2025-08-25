# Intro to Machine Learning

## 1. Overview of AI and Machine Learning

### What is AI?
- **Artificial Intelligence (AI):** Simulates human cognition through computers.
- Encompasses fields like computer vision, NLP, generative AI, and machine learning.

### What is Machine Learning (ML)?
- **ML:** A subset of AI that uses algorithms to learn patterns from data and make predictions or decisions without explicit programming.
- **Deep Learning:** A branch of ML using multi-layered neural networks to automatically extract features from complex data.

---

## 2. Types of Machine Learning

### Supervised Learning
- **Definition:** Learns from labeled data (input-output pairs).
- **Examples:** Classification (spam detection), Regression (house price prediction).

### Unsupervised Learning
- **Definition:** Finds patterns in unlabeled data.
- **Examples:** Clustering (customer segmentation), Dimensionality Reduction.

### Semi-Supervised Learning
- **Definition:** Uses a small amount of labeled data plus a large amount of unlabeled data.
- **Use Case:** When labeling is expensive or time-consuming.

### Reinforcement Learning
- **Definition:** Trains an agent to make decisions by interacting with an environment and receiving feedback (rewards/punishments).
- **Example:** Game AI, robotics.

---

## 3. Key Machine Learning Techniques and Applications

### Core Techniques
- **Classification:** Predicts categories (e.g., spam/ham).
- **Regression:** Predicts continuous values (e.g., temperature).
- **Clustering:** Groups similar data points (e.g., customer segments).
- **Association:** Finds items/events that co-occur (e.g., market basket analysis).
- **Anomaly Detection:** Identifies outliers (e.g., fraud detection).
- **Sequence Mining:** Predicts next events (e.g., website clickstreams).
- **Dimensionality Reduction:** Reduces feature space (e.g., PCA).
- **Recommendation Systems:** Suggests items based on user preferences.

### Real-World Applications
- Disease prediction (e.g., cancer detection)
- Product and content recommendation (e.g., Amazon, Netflix)
- Loan approval and risk modeling (e.g., banks)
- Customer churn prediction (e.g., telecom)
- Computer vision (e.g., image classification)
- Virtual assistants and chatbots

---

## 4. Machine Learning Model Lifecycle

### Main Stages
1. **Problem Definition:** Clearly state the business or technical problem.
2. **Data Collection:** Gather relevant data from various sources.
3. **Data Preparation (ETL):** Clean, transform, and consolidate data.
   - Handle missing values, outliers, data formatting.
   - Create new features if needed.
   - Exploratory Data Analysis (EDA) for pattern discovery.
4. **Model Development:** Build and train ML models using frameworks and libraries.
   - Use pre-existing resources where possible.
   - Select appropriate algorithms based on the problem.
5. **Model Evaluation:** Assess model performance using metrics and test data.
   - Tune hyperparameters, validate with user feedback if possible.
6. **Model Deployment:** Integrate the model into production systems.
   - Monitor and retrain as needed for continuous improvement.

### Practical Tips
- Data cleaning and preparation are often the most time-consuming steps.
- Continuous monitoring post-deployment is critical for maintaining performance.

---

## 5. Tools and Languages for Machine Learning

### Programming Languages
- **Python:** Most popular, extensive libraries (NumPy, Pandas, Scikit-learn, PyTorch, TensorFlow).
- **R:** Great for statistical analysis and data exploration.
- **Others:** Julia (high-performance), Scala (big data pipelines), Java (production), JavaScript (web ML).

### Data Processing & Analytics
- **Databases:** PostgreSQL, Hadoop, Spark, Apache Kafka.
- **Python Libraries:** Pandas (dataframes), NumPy (numerical arrays), SciPy (scientific computing).

### Data Visualization
- **Python:** Matplotlib, Seaborn.
- **R:** ggplot2.
- **Business Intelligence:** Tableau (interactive dashboards).

### Machine Learning & Deep Learning
- **Scikit-learn:** Classical ML algorithms, data preprocessing, model evaluation.
- **TensorFlow, Keras, PyTorch, Theano:** Deep learning frameworks for building neural networks.

### Computer Vision & NLP
- **Computer Vision:** OpenCV, Scikit-Image, TorchVision.
- **NLP:** NLTK, TextBlob, Stanza.

### Generative AI
- **Hugging Face Transformers:** Pre-trained models for NLP tasks.
- **ChatGPT, DALL-E:** Text and image generation.

---

## 6. Scikit-learn: ML Ecosystem Example

### Why Use Scikit-learn?
- Easy to implement ML pipelines in Python.
- Integrates with NumPy, Pandas, SciPy, and Matplotlib.
- Supports classification, regression, clustering, dimensionality reduction.
- Excellent documentation and community support.

### Typical Workflow
1. **Data Preprocessing:** Cleaning, scaling, feature engineering.
2. **Splitting Data:** Train/test split for model validation.
3. **Model Instantiation:** Select and configure ML algorithm (e.g., SVM, Decision Tree).
4. **Training:** Fit the model on training data.
5. **Prediction:** Use the model to predict on new/unseen data.
6. **Evaluation:** Assess accuracy using metrics (e.g., confusion matrix).
7. **Exporting:** Save models for production use.

---

## 7. Backend/AI Engineering: Data Scientist vs. AI Engineer

### Data Scientist
- Focus: Data storytelling, descriptive analytics, EDA, clustering.
- Data: Mostly structured/tabular data, smaller datasets.
- Models: Many narrow, application-specific models.
- Use Cases: Predictive analytics, statistical inference.

### AI Engineer (esp. Generative AI)
- Focus: System building, deploying AI solutions, generative and prescriptive use cases.
- Data: Mainly unstructured (text, images, audio), very large datasets.
- Models: Foundation models (e.g., LLMs) that generalize across tasks.
- Use Cases: Intelligent assistants, chatbots, recommendation engines, automation.
- Methods: Prompt engineering, fine-tuning (PEFT, LORA, QLORA), Retrieval-Augmented Generation (RAG).

### Key Differences
- **Scale:** AI engineers work with much larger models/data.
- **Process:** AI engineers often use pre-trained models and focus on prompt engineering and system integration.
- **Overlap:** Both roles require strong data skills, but AI engineering emphasizes deploying and maintaining AI systems.

---

## 8. Key Takeaways and Actionable Tips

- **ML is about learning from data**—choose your approach based on data type, problem, and resources.
- **Preprocess and explore your data** before modeling.
- **Use the right tools:** Python libraries (Pandas, NumPy, Scikit-learn) are essential for backend/AI work.
- **Understand the ML lifecycle:** Iterative process from problem definition to deployment and monitoring.
- **Stay updated:** The field evolves rapidly—new models, tools, and research emerge frequently.
- **Build a portfolio:** Hands-on projects (e.g., recommendation systems, classification tasks) are crucial for demonstrating skills.
- **Monitor deployed models:** Continuous evaluation and retraining are necessary for real-world impact.

---

## 9. Glossary

- **Feature Engineering:** Creating new input features from raw data to improve model performance.
- **Foundation Model:** Large, general-purpose AI models (e.g., GPT, BERT) that can be adapted for many tasks.
- **Prompt Engineering:** Crafting input prompts to guide the behavior of generative AI models.
- **ETL:** Extract, Transform, Load—a process for data collection and preparation.
- **PEFT/LORA/QLORA:** Techniques for efficient fine-tuning of large models.

---

## 10. Example: End-to-End ML Project (Product Recommendation)

1. **Define Problem:** Recommend products based on user purchase history.
2. **Collect Data:** User transactions, product info, ratings, demographics.
3. **Prepare Data:** Clean, join, create features (e.g., average purchase interval).
4. **Develop Model:** Use content-based and collaborative filtering.
5. **Evaluate:** Test with holdout data, collect user feedback.
6. **Deploy:** Integrate with app, monitor and retrain as needed.

---