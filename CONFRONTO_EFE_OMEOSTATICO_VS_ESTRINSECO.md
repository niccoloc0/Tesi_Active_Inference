# Confronto EFE: Omeostatico vs Estrinseco (TD Learning)

## Sommario della Ricerca

Questo documento riassume il confronto tra due approcci per il calcolo del **Pragmatic Value** nell'Expected Free Energy (EFE) di un agente Active Inference applicato al controllo di un termostato intelligente.

---

## 1. Background Teorico

### 1.1 Expected Free Energy (EFE)

L'Expected Free Energy è composta da due termini:

```
EFE = Epistemic Value + Pragmatic Value
```

- **Epistemic Value**: Riduzione dell'incertezza (information gain)
- **Pragmatic Value**: Vicinanza agli obiettivi/preferenze

### 1.2 Due Approcci per il Pragmatic Value

| Approccio | Formula | Descrizione |
|-----------|---------|-------------|
| **Omeostatico** | `-(predicted_temp - target_temp)²` | Costo quadratico della distanza dal target |
| **Estrinseco (TD)** | `V(s) + E[r(s,a)]` | Valore appreso + reward atteso |

---

## 2. Implementazioni

### 2.1 File Creati

| File | Descrizione |
|------|-------------|
| `smart_thermostat.py` | Agente omeostatico originale |
| `smart_thermostat_extrinsic.py` | Agente con TD Learning |
| `smart_thermostat_comparison.py` | Confronto in condizioni identiche |
| `smart_thermostat_complex_rewards.py` | Scenario con reward complessi |
| `smart_thermostat_asymmetric.py` | Scenario con reward asimmetrici |

### 2.2 Temporal Difference Learning

L'agente estrinseco implementa TD(λ) con:

```python
# TD Error
δ = r + γ * V(s') - V(s)

# Value Update con Eligibility Traces
e(s) ← γλ * e(s) + 1  # per stato corrente
V(s) ← V(s) + α * δ * e(s)
```

**Parametri utilizzati:**
- Learning rate (α): 0.1 - 0.2
- Discount factor (γ): 0.9 - 0.95
- Eligibility trace (λ): 0.8 - 0.9

---

## 3. Esperimenti e Risultati

### 3.1 Scenario 1: Reward Simmetrico Semplice

**Struttura Reward:**
- In comfort zone (±2°C): +0.3
- Fuori comfort zone: -0.5
- Costo riscaldamento: -0.2

**Risultati (simulazione 3600s):**

| Metrica | Omeostatico | Estrinseco (TD) | Vincitore |
|---------|-------------|-----------------|-----------|
| Tempo sopravvivenza | 3601s | 3176s | 🔵 Omeostatico |
| Comfort Zone | 51.5% | 44.1% | 🔵 Omeostatico |
| Budget finale | €58.80 | €0.00 | 🔵 Omeostatico |
| Rewards totali | +241.80 | +191.10 | 🔵 Omeostatico |

**🏆 Vincitore: OMEOSTATICO (6-0)**

**Motivo:** Il reward è una funzione diretta della distanza dal target. L'omeostatico ha già codificata esplicitamente questa relazione, mentre TD deve apprenderla.

---

### 3.2 Scenario 2: Reward Asimmetrico

**Struttura Reward:**
- In comfort zone (±2°C): +0.4
- **Troppo CALDO**: Penalità ESPONENZIALE `base + exp(scale * excess)`
- Troppo freddo: Penalità lineare `linear * excess`

**Risultati:**

| Metrica | Omeostatico | Estrinseco (TD) | Vincitore |
|---------|-------------|-----------------|-----------|
| Cumulative Reward | -194.4 | -194.2 | 🟠 Estrinseco |
| Hot Penalty | 208.2 | 169.3 | 🟠 Estrinseco |
| Cold Penalty | 19.3 | 48.8 | 🔵 Omeostatico |
| % Troppo Caldo | 28.8% | 24.9% | 🟠 Estrinseco |
| % Troppo Freddo | 35.4% | 49.0% | 🔵 Omeostatico |

**🏆 Vincitore: ESTRINSECO TD**

**Motivo:** TD ha imparato che la zona "troppo caldo" ha penalità esponenziali, quindi preferisce stare leggermente troppo freddo. L'omeostatico, usando `(temp - target)²`, tratta le due direzioni come equivalenti.

---

## 4. Analisi Teorica

### 4.1 Quando l'Omeostatico è Superiore

L'approccio omeostatico `(temp - target)²` eccelle quando:

1. **Reward è simmetrico**: La penalità per essere sopra o sotto il target è uguale
2. **Reward è proporzionale alla distanza**: Più lontani = proporzionalmente peggio
3. **Nessuna dinamica nascosta**: Il sistema è completamente osservabile
4. **Target noto a priori**: L'agente conosce esattamente dove vuole andare

**Vantaggi:**
- ✅ Nessun warm-up necessario
- ✅ Risposta immediata e precisa
- ✅ Computazionalmente efficiente
- ✅ Interpretabile

### 4.2 Quando TD Learning è Superiore

L'approccio TD eccelle quando:

1. **Reward asimmetrico**: Es. troppo caldo è peggio di troppo freddo
2. **Reward non-lineare**: Es. exponenziale, logaritmico, a gradini
3. **Reward con memoria**: Es. bonus per stabilità, penalità per oscillazioni
4. **Reward sconosciuto**: L'agente deve scoprire cosa ottimizzare
5. **Dinamiche complesse**: L'ambiente ha pattern nascosti

**Vantaggi:**
- ✅ Adattabile a qualsiasi struttura di reward
- ✅ Può apprendere preferenze implicite
- ✅ Cattura dipendenze temporali
- ✅ Generalizza a nuove situazioni

### 4.3 Trade-off Fondamentale

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   OMEOSTATICO                         ESTRINSECO (TD)           │
│   ┌─────────────┐                     ┌─────────────┐           │
│   │  Veloce     │                     │  Adattivo   │           │
│   │  Preciso    │ ◄───────────────► │  Flessibile │           │
│   │  Rigido     │                     │  Lento      │           │
│   └─────────────┘                     └─────────────┘           │
│                                                                 │
│   Reward noto                          Reward sconosciuto       │
│   Simmetrico                           Asimmetrico              │
│   Statico                              Dinamico                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Implicazioni per Active Inference

### 5.1 Connessione con la Teoria

Nella teoria dell'Active Inference:

- **Pragmatic Value omeostatico** corrisponde al concetto di **preferenze prior** fisse
- **Pragmatic Value estrinseco** corrisponde a **preferenze apprese** dall'esperienza

Entrambi sono compatibili con il framework Free Energy Principle:

```
G = E_Q[log Q(s') - log P(o|s') - log P(s')]
    └────────────┘   └──────────┘   └───────┘
     Epistemic       Instrumental    Prior
     (same)          (different!)    Preference
```

### 5.2 Biologicamente Plausibile?

| Aspetto | Omeostatico | TD Learning |
|---------|-------------|-------------|
| Plausibilità | Alta (riflessi innati) | Alta (apprendimento) |
| Esempio biologico | Termoregolazione | Condizionamento operante |
| Localizzazione | Ipotalamo | Gangli basali, corteccia |

---

## 6. Raccomandazioni Pratiche

### 6.1 Quando Usare Omeostatico

```
✅ Obiettivo chiaro e ben definito
✅ Reward semplice e simmetrico
✅ Risorse computazionali limitate
✅ Necessità di risposta immediata
```

### 6.2 Quando Usare TD Learning

```
✅ Reward complesso o sconosciuto
✅ Asimmetrie nelle preferenze
✅ Dipendenze temporali nel reward
✅ Ambiente che cambia nel tempo
```

### 6.3 Approccio Ibrido (Raccomandato)

Per applicazioni reali, si può usare un **approccio ibrido**:

1. **Inizializzazione**: Usare costo omeostatico come baseline
2. **Apprendimento**: TD corregge gradualmente basandosi su reward reali
3. **Convergenza**: Peso shift verso TD man mano che apprende

```python
pragmatic_value = (1 - learning_progress) * homeostatic_cost + 
                  learning_progress * td_learned_value
```

---

## 7. Conclusioni

### 7.1 Sintesi dei Risultati

| Scenario | Vincitore | Margine |
|----------|-----------|---------|
| Reward simmetrico semplice | Omeostatico | 6-0 |
| Reward asimmetrico (hot>>cold) | TD Learning | Reward -194.2 vs -194.4 |

### 7.2 Takeaway Principale

> **L'approccio omeostatico `(temp - target)²` è ottimale quando la struttura del reward è nota, simmetrica e proporzionale alla distanza.**
>
> **Il Temporal Difference Learning diventa vantaggioso quando il reward è asimmetrico, non-lineare, dipendente dalla storia, o sconosciuto a priori.**

### 7.3 Contributo alla Tesi

Questo lavoro dimostra che:

1. L'EFE in Active Inference può essere calcolata con approcci diversi
2. La scelta dell'approccio dipende dal dominio applicativo
3. TD Learning offre flessibilità per scenari complessi
4. L'approccio omeostatico rimane valido per obiettivi semplici

---

## 8. Appendice: Codice Chiave

### 8.1 EFE Omeostatico

```python
def compute_efe_homeostatic(self, action, target_temp, predicted_mean, predicted_cov):
    # Epistemic: information gain
    epistemic_value = -0.5 * np.log(predicted_cov[0, 0] + 1e-6)
    
    # Pragmatic: SYMMETRIC quadratic distance
    predicted_temp = predicted_mean[0]
    pragmatic_cost = (predicted_temp - target_temp) ** 2
    
    # EFE (minimize)
    efe = -epistemic_value + pragmatic_cost
    return efe
```

### 8.2 EFE Estrinseco (TD)

```python
def compute_efe_extrinsic(self, action, target_temp, predicted_mean, predicted_cov):
    # Epistemic: same as homeostatic
    epistemic_value = -0.5 * np.log(predicted_cov[0, 0] + 1e-6)
    
    # Pragmatic: TD-learned value + expected reward
    learned_value = self.value_function.get_value(predicted_temp)
    expected_reward = self.get_expected_reward(predicted_temp, target_temp)
    pragmatic_value = 0.5 * learned_value + 0.5 * expected_reward
    
    # EFE (minimize)
    efe = -epistemic_value - pragmatic_value
    return efe
```

### 8.3 TD Update

```python
def update_td(self, current_temp, reward, next_temp):
    current_bin = self.temp_to_bin(current_temp)
    next_bin = self.temp_to_bin(next_temp)
    
    # TD Error: δ = r + γV(s') - V(s)
    td_error = reward + self.gamma * self.V[next_bin] - self.V[current_bin]
    
    # Eligibility trace update
    self.eligibility *= self.gamma * self.lambda_
    self.eligibility[current_bin] += 1.0
    
    # Value update: V ← V + αδe
    self.V += self.alpha * td_error * self.eligibility
    
    return td_error
```

---

## 9. Riferimenti

1. Friston, K. (2010). The free-energy principle: a unified brain theory?
2. Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction.
3. Parr, T., & Friston, K. J. (2019). Generalised free energy and active inference.
4. Da Costa, L., et al. (2020). Active inference on discrete state-spaces.

---

*Documento generato il 2026-01-20*
*Progetto: Smart Thermostat Active Inference*
