# 🛍️ Customer Segmentation Using E-Commerce Data

This project performs customer segmentation using **RFM (Recency, Frequency, Monetary)** analysis and **KMeans Clustering** on an online retail dataset. By analyzing customer purchasing behavior, it groups customers into distinct segments to support targeted marketing and decision-making.

---

## 📌 Features

- 📅 **Recency**: How recently a customer made a purchase  
- 🔁 **Frequency**: How often they purchase  
- 💰 **Monetary**: How much they spent  

- 🧹 Data Cleaning and Preprocessing  
- 📊 RFM Table Creation  
- 📈 Data Scaling using `StandardScaler`  
- 🤖 KMeans Clustering (`sklearn`)  
- 📉 Dimensionality Reduction using PCA for 2D Visualization  
- 📦 Export of Segmented Customer Data (`customer_segments.csv`)  

---

## 📁 Dataset

This project uses the [Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail) from the UCI Machine Learning Repository.

Make sure the dataset file is named:  

OnlineRetail.csv


And placed in the same directory or update the path accordingly in the script.

---

## 🔧 Installation

Install required Python libraries:
pip install pandas numpy matplotlib seaborn scikit-learn


**##🚀How to Run**

Clone this repository

Place OnlineRetail.csv in the project folder

Run the script:

python customer_segmentation.py


The script will:

Clean the data

Generate RFM metrics

Perform KMeans clustering

Plot RFM cluster statistics and 2D PCA scatter plot

Save the segmented data to customer_segments.csv

**📊 Example Output**

🎯 Cluster Summary Table:
Shows average Recency, Frequency, and Monetary values per cluster.

📦 Visualizations:
Boxplots of Recency, Frequency, and Monetary by cluster

2D scatter plot (PCA) with color-coded clusters

📁 Output
customer_segments.csv
A CSV file containing CustomerID, RFM values, and cluster labels.

🧠 Use Cases
Targeted marketing campaigns

Loyalty programs

Customer retention strategies

Business intelligence dashboards

👤 Author
Tufan
📍 Built with Python, Pandas, Scikit-learn, and Matplotlib

📜 License
This project is licensed under the MIT License. Feel free to use and modify.

📬 Contributions
Issues and pull requests are welcome! Open an issue for suggestions or improvements.
