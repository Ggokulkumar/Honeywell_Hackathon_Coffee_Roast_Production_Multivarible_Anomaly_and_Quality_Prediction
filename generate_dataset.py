import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_coffee_roast_dataset(n_batches=10000, anomaly_rate=0.20):
    np.random.seed(42)
    random.seed(42)
    
    data = []
    
    bean_types = ['Arabica', 'Robusta', 'Blend']
    roast_levels = ['Light', 'Medium', 'Dark']
    
    for batch_id in range(1, n_batches + 1):
        bean_type = random.choice(bean_types)
        target_roast = random.choice(roast_levels)
        batch_size = np.random.normal(100, 15) 
        
        initial_moisture = np.random.normal(11.5, 1.5)  
        bean_density = np.random.normal(1.3, 0.15)   
        bean_size = np.random.normal(6.5, 0.8)      
        
        preheat_temp = np.random.normal(205, 15)    
        gas_level = np.random.normal(75, 12)        
        airflow_rate = np.random.normal(80, 10)      
        drum_speed = np.random.normal(65, 8)         
        
        if target_roast == 'Light':
            roast_duration = np.random.normal(12, 1.5)
        elif target_roast == 'Medium':
            roast_duration = np.random.normal(14, 1.5)
        else:
            roast_duration = np.random.normal(16, 2)
            
        final_temp = 0
        if target_roast == 'Light':
            final_temp = np.random.normal(205, 5)
        elif target_roast == 'Medium':
            final_temp = np.random.normal(220, 5)
        else:
            final_temp = np.random.normal(235, 5)
            
        first_crack_time = np.random.normal(roast_duration * 0.6, 1)
        second_crack_time = np.random.normal(roast_duration * 0.85, 1) if target_roast != 'Light' else None
        
        ambient_temp = np.random.normal(25, 8)
        humidity = np.random.normal(60, 20)
        
        anomaly = 0
        anomaly_type = "None"
        
        if random.random() < anomaly_rate:
            choice = random.random()
            if choice < 0.25: 
                anomaly = 1
                anomaly_type = "Scorching"
                gas_level = np.random.uniform(90, 105)
                airflow_rate = np.random.uniform(50, 70)
            elif choice < 0.5: 
                anomaly = 1
                anomaly_type = "Baking"
                gas_level = np.random.uniform(40, 60)
                roast_duration *= 1.3 
            elif choice < 0.75: 
                anomaly = 1
                anomaly_type = "Underdeveloped"
                roast_duration *= 0.7 
                final_temp -= 15
            else: 
                anomaly = 1
                anomaly_type = "Tipping"
                preheat_temp = np.random.uniform(220, 240)
                drum_speed = np.random.uniform(80, 95)
                
        heat_rate = (final_temp - preheat_temp) / roast_duration if roast_duration > 0 else 0
        total_energy = gas_level * roast_duration * batch_size / 100
        
        quality_score = 10.0
        
        quality_score -= abs(gas_level - 75)**1.5 * 0.005
        quality_score -= abs(airflow_rate - 80)**1.5 * 0.004
        if target_roast == 'Light':
            quality_score -= abs(final_temp - 205) * 0.2
        elif target_roast == 'Medium':
            quality_score -= abs(final_temp - 220) * 0.2
        else:
            quality_score -= abs(final_temp - 235) * 0.2

        if gas_level > 90 and airflow_rate < 70:
            quality_score -= 1.5
            
        if anomaly_type == "Scorching":
            quality_score -= np.random.uniform(2.5, 4.0)
        elif anomaly_type == "Baking":
            quality_score -= np.random.uniform(2.0, 3.5)
        elif anomaly_type == "Underdeveloped":
            quality_score -= np.random.uniform(3.0, 4.5)
        
        quality_score += np.random.normal(0, 0.2)
        quality_score = np.clip(quality_score, 1.0, 10.0)

        defects = np.random.poisson(max(0, 10 - quality_score) * 0.5)
        aroma_score = np.clip(quality_score - np.random.uniform(0.5, 1.5), 1, 10)
        body_score = np.clip(quality_score - np.random.uniform(0.5, 1.5), 1, 10)
        acidity_score = np.clip(quality_score - np.random.uniform(0.5, 1.5), 1, 10)
        color_score = 60 - (final_temp - 215) * 2 + np.random.normal(0, 3)

        weight_loss = 15 + (final_temp - 220) * 0.2 + (roast_duration - 14) * 0.3
        final_weight = batch_size * (1 - weight_loss/100)
        
        batch_data = {
            'batch_id': batch_id,
            'timestamp': datetime.now() - timedelta(days=n_batches-batch_id),
            'bean_type': bean_type,
            'target_roast_level': target_roast,
            'batch_size_kg': round(batch_size, 2),
            'initial_moisture_pct': round(initial_moisture, 2),
            'bean_density_g_cm3': round(bean_density, 3),
            'bean_size_mm': round(bean_size, 2),
            'preheat_temp_c': round(preheat_temp, 1),
            'gas_level_pct': round(gas_level, 1),
            'airflow_rate_pct': round(airflow_rate, 1),
            'drum_speed_rpm': round(drum_speed, 1),
            'roast_duration_min': round(roast_duration, 2),
            'drying_temp_avg_c': round(preheat_temp + (final_temp - preheat_temp) * 0.2, 1),
            'maillard_temp_avg_c': round(preheat_temp + (final_temp - preheat_temp) * 0.5, 1),
            'development_temp_avg_c': round(preheat_temp + (final_temp - preheat_temp) * 0.8, 1),
            'final_temp_c': round(final_temp, 1),
            'first_crack_time_min': round(first_crack_time, 2),
            'second_crack_time_min': round(second_crack_time, 2) if second_crack_time else None,
            'ambient_temp_c': round(ambient_temp, 1),
            'humidity_pct': round(humidity, 1),
            'heat_rate_c_per_min': round(heat_rate, 2),
            'total_energy_units': round(total_energy, 1),
            'weight_loss_pct': round(weight_loss, 2),
            'final_weight_kg': round(final_weight, 2),
            'color_score_agtron': round(color_score, 1),
            'aroma_score': round(aroma_score, 2),
            'body_score': round(body_score, 2),
            'acidity_score': round(acidity_score, 2),
            'overall_quality_score': round(quality_score, 2),
            'defects_count': defects,
            'quality_pass': 1 if quality_score >= 7.0 and defects <= 2 else 0,
            'process_anomaly': anomaly
        }
        
        data.append(batch_data)
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("Generating enhanced coffee roasting dataset...")
    dataset = generate_coffee_roast_dataset(n_batches=10000)

    dataset.to_csv('FNB_Coffee_Roast_Dataset.csv', index=False)
    print(f"Dataset saved with {len(dataset)} batches and {len(dataset.columns)} features.")

    print("\n--- Dataset Overview ---")
    print(f"Shape: {dataset.shape}")
    print(f"\nQuality Pass Distribution:")
    print(dataset['quality_pass'].value_counts(normalize=True))
    print(f"\nAnomaly Distribution:")
    print(dataset['process_anomaly'].value_counts(normalize=True))

    print("\n--- Sample Data ---")
    print(dataset.head())

    numeric_cols = dataset.select_dtypes(include=[np.number]).columns
    quality_corr = dataset[numeric_cols].corr()['overall_quality_score'].abs().sort_values(ascending=False)
    print(f"\n--- Top 10 Features Most Correlated with Quality ---")
    print(quality_corr.head(10))
