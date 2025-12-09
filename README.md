# Smart Thermostat - Tesi sull'Active Inference

Questo progetto implementa un sistema di controllo della temperatura che utilizza l'**Inferenza Attiva** (Active Inference) per prendere decisioni ottimali. Il sistema è composto da due componenti principali che comunicano in modo sicuro attraverso code di messaggi.

## Architettura

```
┌───────────────────────────────────────────────────────────────────┐
│                        SMART THERMOSTAT                           │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌──────────────┐    Queue (HMAC)    ┌──────────────────────┐    │
│   │              │ ──────────────────▶│                      │    │
│   │    SENSOR    │                    │   ACTIVE INFERENCE   │    │
│   │              │◀────────────────── │        AGENT         │    │
│   └──────────────┘    Queue (HMAC)    └──────────────────────┘    │
│         │                                       │                 │
│         ▼                                       ▼                 │
│   ┌──────────────┐                     ┌──────────────────────┐   │
│   │   Ambiente   │                     │   Filtro di Kalman   │   │
│   │   Fisico     │                     │   Free Energy        │   │
│   └──────────────┘                     └──────────────────────┘   │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

## Agente

L'agente implementa il framework dell'**Inferenza Attiva**, che combina:

### 1. Modello generativo (Filtro di Kalman)
- **Stato**: `[temperatura, tasso_variazione_temperatura]`
- **Predizione**: Prevede la temperatura futura basandosi sullo stato attuale e le azioni
- **Aggiornamento**: Corregge le credenze usando le osservazioni del sensore

### 2. Minimizzazione dell'energia libera
L'agente minimizza l'**energia libera** per:
- Ridurre l'**errore di predizione** tra aspettative e osservazioni
- Mantenere la temperatura vicina al target stagionale

### 3. Expected Free Energy (EFE)
Per selezionare le azioni, l'agente calcola:
- **Valore epistemico**: Guadagno informativo (riduzione dell'incertezza)
- **Valore pragmatico**: Vicinanza al target desiderato

```python
efe = -valore_epistemico + costo_pragmatico
```

### 4. Azioni epistemiche
L'agente adatta il **sampling rate** in base all'incertezza:
- Alta incertezza → Campionamento frequente (dt basso)
- Bassa incertezza → Campionamento raro (dt alto)

### 5. Apprendimento online
L'agente impara durante l'esecuzione:
- **Precisione delle osservazioni**: Quanto sono affidabili i sensori
- **Effetto del riscaldamento**: Quanto la temperatura aumenta quando il riscaldatore è acceso

## Sensor

Il sensore simula l'**ambiente fisico**:

### Dinamiche termiche
```python
temperatura += calore_riscaldatore + scambio_termico_esterno
```

### Stagioni
| Stagione   | Temp. Esterna | Target Interno |
|------------|---------------|----------------|
| Inverno    | 0°C           | 20.0°C         |
| Primavera  | 15°C          | 21.0°C         |
| Estate     | 35°C          | 24.0°C         |
| Autunno    | 12°C          | 20.5°C         |

### Logging con WandB
Tutte le metriche vengono registrate tramite [Weights & Biases](https://wandb.ai) per l'analisi.

## Sicurezza HMAC

La comunicazione tra sensore e agente è protetta tramite **HMAC-SHA256**:

```python
class SecureMessage:
    SECRET_KEY = b"!T\q!Un'8AL4bpHH"
    
    # Firma un messaggio
    signed = SecureMessage.sign_message(data)
    
    # Verifica un messaggio
    is_valid, payload = SecureMessage.verify_message(signed)
```

### Caratteristiche di sicurezza
- **Integrità**: I messaggi non possono essere modificati senza invalidare la firma
- **Autenticità**: Solo chi conosce la chiave segreta può creare firme valide
- **Protezione timing attack**: Usa `hmac.compare_digest()` per confronti sicuri

## Metriche monitorate

| Metrica | Descrizione |
|---------|------------|
| Free energy | Energia libera variazionale |
| Prediction error | Errore tra predizione e osservazione |
| Epistemic value | Valore informativo dell'azione |
| Pragmatic value | Valore pratico (vicinanza al target) |
| Uncertainty | Varianza della credenza sulla temperatura |
| Belief mean | Media della credenza sulla temperatura |
| Agent budget | Budget rimanente dell'agente |

## Esecuzione

### Requisiti
```bash
pip install numpy wandb
```

### Avviare la Simulazione
```bash
python3 smart_thermostat.py
```

### Output Esempio
```
[Agent] Active Inference initialized with budget: 75.00
[Agent] Initial belief: temp=19.00°C, uncertainty=4.00

[Agent] Step 0 | Season: Inverno | Budget: 74.5
  Observation: 18.50°C | Belief: 18.45°C | Error: 0.05°C
  Uncertainty: 0.800 | Free Energy: 1.234
  Epistemic Value: -0.500 | Pragmatic Value: -2.250
  Action: TURN_ON | Heater: ON | Next dt: 2.0s
```

## Configurazione

Modifica i parametri in `wandb.init()`:

```python
config={
    "simulation_duration": 720,    # Durata simulazione (secondi)
    "steps_per_season": 60,        # Step per stagione
    "heating_power": 1.5,          # Potenza riscaldamento
    "weather_fluctuation": 1.5,    # Fluttuazione meteo
}
```

## Struttura file

```
Tesi_Active_Inference/
├── smart_thermostat.py    # Simulazione principale
├── README.md              # Documentazione
└── wandb/                 # Log delle esecuzioni
```

## Risultati

La simulazione termina quando:
1. Il **budget dell'agente si esaurisce** (controllo dei costi)
2. Si raggiunge la **durata massima** della simulazione

I risultati sono visualizzabili su WandB con grafici interattivi delle metriche.

## Teoria

### 1. Cos'è l'Active Inference?

L’**Active Inference** è un framework ispirato al **principio di energia libera**, utilizzato in neuroscienze e intelligenza artificiale per modellare il comportamento di agenti che bilanciano **esplorazione** (riduzione dell’incertezza) ed **esecuzione** (raggiungimento di obiettivi).

In questo contesto, ogni agente mantiene una **distribuzione di probabilità** (ad esempio una “curva a campana”) sulla fiducia in un sensore, aggiornata in base alle osservazioni.

Invece di decidere in modo deterministico, l’agente utilizza l’**Expected Free Energy (EFE)** per scegliere le azioni:
- Se l’incertezza è elevata, **verifica il sensore**.
- Se invece l’obiettivo (ad esempio fornire dati in tempo reale) è urgente, **usa i dati con cautela**.

Questo approccio consente di **evolvere dinamicamente la fiducia** nel tempo, rendendo il sistema più **resiliente ad attacchi o errori**.

### 2. Cos'è il filtro di Kalman?

Il **filtro di Kalman** è un efficiente filtro ricorsivo che valuta lo stato di un **sistema dinamico** a partire da una serie di misure soggette a rumore. Per le sue caratteristiche intrinseche, è un filtro ottimo per rumori e disturbi agenti su sistemi gaussiani a media nulla. Trova utilizzo come osservatore dello stato, come **loop transfer recovery (LTR)** e come sistema di identificazione parametrica. Il problema della progettazione del filtro di Kalman è il problema duale del **regolatore lineare quadratico (LQR)**.

### 2. Cos'è l'HMAC?

HMAC (**Keyed-Hash Message Authentication Code** o **Hash-based Message Authentication Code**) è un meccanismo per l'autenticazione di messaggi basato su una funzione di hash.
Tramite HMAC è possibile garantire sia l'**integrità** che l'**autenticità** di un messaggio. HMAC non si occupa della cifratura: il messaggio (crittografato o meno) deve essere trasmesso insieme al codice HMAC. I destinatari in possesso della **chiave segreta** applicano l'algoritmo al messaggio ricevuto e verificano che il codice calcolato corrisponda a quello ricevuto; se sono identici, il messaggio è autentico. HMAC utilizza infatti una combinazione del messaggio originale e della **chiave segreta** per generare il codice.
Una caratteristica peculiare di HMAC è l'indipendenza dalla specifica funzione di hash utilizzata; ciò rende possibile la sostituzione della funzione qualora questa non fosse più abbastanza sicura. Storicamente le funzioni più utilizzate sono state **MD5** e **SHA-1**, sebbene entrambe siano attualmente considerate poco sicure.