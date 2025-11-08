# Tesi su Active Inference

## Cos'è l'Active Inference?

L’**Active Inference** è un framework ispirato al **principio di energia libera**, utilizzato in neuroscienze e intelligenza artificiale per modellare il comportamento di agenti che bilanciano **esplorazione** (riduzione dell’incertezza) ed **esecuzione** (raggiungimento di obiettivi).

In questo contesto, ogni agente mantiene una **distribuzione di probabilità** (ad esempio una “curva a campana”) sulla fiducia in un sensore, aggiornata in base alle osservazioni.

Invece di decidere in modo deterministico, l’agente utilizza l’**Expected Free Energy (EFE)** per scegliere le azioni:
- Se l’incertezza è elevata, **verifica il sensore**.
- Se invece l’obiettivo (ad esempio fornire dati in tempo reale) è urgente, **usa i dati con cautela**.

Questo approccio consente di **evolvere dinamicamente la fiducia** nel tempo, rendendo il sistema più **resiliente ad attacchi o errori**.
