import wandb
import random
import time

# Inizializzazione per avere esecuzioni sempre diverse
random.seed()

# --- 1. Inizializzazione di Weights & Biases ---
run = wandb.init(
    project="Smart Thermostat Simulation",
    config={
        "simulation_steps": 240,
        "steps_per_season": 60,
        "anomaly_prob_on": 0.03,
        "anomaly_prob_off": 0.02,
        "heating_power": 1.5,
        "cooling_factor": 0.5,
        "comfort_margin": 1.0,
        "weather_fluctuation": 1.5,
        "sensor_noise": 0.25 
    }
)

# --- 2. Configurazione dell'environment ---
SEASONS = {
    "Inverno": {"outside_temp": 0,  "target_temp": 20.0},
    "Primavera": {"outside_temp": 15, "target_temp": 21.0},
    "Estate":    {"outside_temp": 35, "target_temp": 24.0},
    "Autunno": {"outside_temp": 12, "target_temp": 20.5}
}
season_order = ["Inverno", "Primavera", "Estate", "Autunno"]

# Stato iniziale random
room_temperature = random.uniform(17.5, 19.5)
heater_on = False
current_season_index = 0

print(f"Simulazione realistica avviata... Temp iniziale: {room_temperature:.1f}°C. Controlla il link di wandb.")

# --- 3. Ciclo di Simulazione ---
for step in range(wandb.config["simulation_steps"]):
    # --- A. AGGIORNAMENTO DELL'ENVIRONMENT ---
    
    if step % wandb.config["steps_per_season"] == 0 and step > 0:
        current_season_index = (current_season_index + 1) % len(season_order)
    
    current_season_name = season_order[current_season_index]
    season_config = SEASONS[current_season_name]
    
    # MODIFICA: Aggiungiamo fluttuazioni meteo realistiche
    base_outside_temp = season_config["outside_temp"]
    weather_effect = random.uniform(-wandb.config["weather_fluctuation"], wandb.config["weather_fluctuation"])
    outside_temp = base_outside_temp + weather_effect
    
    target_temp = season_config["target_temp"]

    # Le anomalie continuano ad essere casuali
    anomaly_type_occurred = "None"
    if not heater_on and random.random() < wandb.config["anomaly_prob_on"]:
        heater_on = True
        anomaly_type_occurred = "Stuck On"
    elif heater_on and random.random() < wandb.config["anomaly_prob_off"]:
        heater_on = False
        anomaly_type_occurred = "Stuck Off"

    # --- B. SENSORE E AGENTE ---
    
    # Il sensore ha un margine di errore (rumore)
    sensed_temperature = room_temperature + random.uniform(-wandb.config["sensor_noise"], wandb.config["sensor_noise"])
    
    action_taken = "IDLE"
    summer_anomaly_corrected = 0
    winter_anomaly_corrected = 0

    # La logica dell'agente ora si basa sulla temperatura *percepita*, non su quella reale
    if current_season_name == "Estate" and heater_on:
        heater_on = False
        action_taken = "EMERGENCY_SHUTDOWN"
        summer_anomaly_corrected = 1
    
    elif current_season_name == "Inverno" and not heater_on and sensed_temperature < target_temp - wandb.config["comfort_margin"]:
        heater_on = True
        action_taken = "FORCED_RESTART"
        if anomaly_type_occurred == "Stuck Off":
             winter_anomaly_corrected = 1

    else:
        if sensed_temperature < target_temp - wandb.config["comfort_margin"]:
            if not heater_on:
                heater_on = True
                action_taken = "TURN_ON"
        
        elif sensed_temperature > target_temp + wandb.config["comfort_margin"]:
            if heater_on:
                heater_on = False
                action_taken = "TURN_OFF"

    # --- C. AGGIORNAMENTO DELLO STATO FISICO (reale) DELLA STANZA ---
    heat_from_heater = wandb.config["heating_power"] if heater_on else 0
    heat_loss_or_gain = (outside_temp - room_temperature) * wandb.config["cooling_factor"]
    # La temperatura REALE viene aggiornata
    room_temperature += heat_from_heater + heat_loss_or_gain
    
    # --- D. LOGGING con W&B ---
    wandb.log({
        "step": step,
        "Stagione": current_season_index,
        "Temperatura Stanza (Reale)": room_temperature,
        "Temperatura (Percepita dal Sensore)": sensed_temperature,
        "Temperatura Target": target_temp,
        "Temperatura Esterna (con Meteo)": outside_temp,
        "Termosifone Acceso": 1 if heater_on else 0,
        "Correzione Anomalia Estiva": summer_anomaly_corrected,
        "Correzione Anomalia Invernale": winter_anomaly_corrected,
    })
    
    # ... (stampa a terminale omessa per brevità) ...
    time.sleep(0.05) # Accorciamo un po' il tempo per velocizzare

# --- 4. Fine della simulazione ---
wandb.finish()
print("\nSimulazione completata.")