# Wallet-Score-Analyser-using-NNClustering

Important links:
Output Csv

1) Google Sheet: https://docs.google.com/spreadsheets/d/10ZEB6OuFd2dg6tJNI1xBHjHwisuW8SZaQ-5ztKZmDEo/edit?usp=sharing
2) .csv extension file: https://drive.google.com/file/d/1Ylp_sEm8p3-0TkDmYQru32Z97wDbKci-/view?usp=sharing

Kaggle Notebook Link: https://www.kaggle.com/code/aayushmancodez/zeru-eth-simple-version

**Process Summary:**
This model utilises Neural Networks Clustering and further analysis of data based on given metrics defined below respectively to assign credit scores and analyse data.

_Proces:_

In the Data Ingestion process the system scans the provided directory to identify the 3 largest JSON files containing Compound V2 transaction data (typically >100MB each) and then it implements error handling for corrupted files or invalid JSON structures. For tiimestamps,  code converts UNIX timestamps (e.g., "1726203335") to pandas DateTime objects using pd.to_datetime(unit='s'). The Feature Engineering part creates comprehensive wallet-level features (temporal patterns, financial metrics, risk indicators). The code makes use of Neural Network which uses an autoencoder for dimensionality reduction before clustering using K-Means Clustering to produce cluster output.

This credit scoring system analyzes wallet activity across several key factors to assign scores from 0 to 100. Higher scores go to wallets that show:

1.	Strong transaction volume (40% weight) – Not just one time spikes, but consistent activity over time this shows sustained usage.
2.	Stability (30%) – Measured by how evenly distributed transactions are.
3.	Diverse actions (20%) – Wallets that engage in multiple transaction types (deposits, withdrawals, etc.) score better than those with repetitive behaviour.
4.	Low risk (10%) – Penalties apply for liquidation events, extreme volatility, or unusually large transactions.
The system also looks at timing patterns—like how often a wallet is active and whether transactions follow predictable rhythms—to further refine scores.

Before final scorring, the data goes through a neural network to simplify complex patterns, followed by clustering (grouping similar wallets) and anomaly detection (pointing out suspicious behaviour, which reduces scores by 30%).
Final scores are grouped into tiers:

•	0-40: High-risk (possible bots or exploiters)

•	41-65: Moderate-risk (needs review)

•	66-85: Reliable (mostly safe)

•	86-100: Excellent (consistent, diverse, low-risk activity)

The system divides wallets into groups/tiers as shown above, ensuring fairness based on actual behaviour rather than random cutoffs. 




In depth Part by Part explanation:
**Temporal Behaviour Features explained in depth:**

1)Activity Patterns:

Time Entropy: Measures unpredictability of transaction timing (-Σp*log(p))
Periodicity: Spectral analysis of transaction intervals using FFT
Autocorrelation: 1-day and 7-day self-similarity metrics
2) Timing Statistics:

Skew/Kurtosis: Distribution shape of time-between-transactions
Active Hours: Count of distinct hours with activity (0-24)
Recency: Exponential decay score based on days since last activity

Financial Profile Features:

Volume Metrics: Total USD volume (log-scaled), Average transaction size and Volatility (std/mean ratio)
Risk Indicators: Liquidation ratio (# liquidations/total txs), Large transaction frequency (>10x mean amount)  and MAD (Median Absolute Deviation) of amounts

**Neural Network Preprocessing:**

Code used:

    def create_autoencoder(input_dim=31, encoding_dim=32):
    
    input_layer = Input(shape=(input_dim,))
    
    encoder = Dense(encoding_dim, activation='relu')(input_layer) 
    
    decoder = Dense(input_dim, activation='linear')(encoder)
    
    return Model(inputs=input_layer, outputs=decoder)
    
- Trained for 100 epochs with Adam optimizer (lr=0.001)
- Bottleneck layer forces learning of compressed representation
- Reconstruction loss (MSE) typically converges to 0.12-0.15

**Clustering Layer:**

Processes 32D encoded features with KMeans (k=10)
Optimal cluster count determined by elbow method

3. Anomaly Detection:
Local Outlier Factor (LOF) with k=20 neighbors
Contamination set to 10% based on protocol history
Anomaly score reduces final credit score by 30%
Scoring Algorithm

**Base Score Components:**

Score = 0.4*(volume_norm) + 0.3*(stability) + 0.2*(diversity) + 0.1*(1-risk)

Where:

volume_norm = log10(total_volume) / max(log_volume)

stability = 1/(1 + amount_std)

diversity = action_type_entropy

risk = 0.4*liquidation_ratio + 0.3*volatility + 0.2*large_tx_ratio + 0.1*autocorrelation


