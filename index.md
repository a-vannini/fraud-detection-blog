---
layout: default
---

![Banner](assets/banner.png)

<!-- [IBM Datensatz Money Laundering](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml). -->

<!-- <img src="assets/eda_tage.png" alt="Example" style="float: left; margin-right: 20px; width: 300px;"> -->
<!-- <img src="assets/eda_tage.png" alt="Example" style="float: right; margin-left: 20px; width: 300px;"> -->
<!-- [IBM Transactions for Anti Money Laundering (AML)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) -->


Dieser Blogbeitrag bietet Einblicke in die Erkennung von Geldwäscheversuchen mit Machine Learning. Wir stellen verschiedene Ansätze vor und teilen unsere Erfahrungen. Der Schwerpunkt liegt darauf, die ersten Schritte von klassischen Machine-Learning-Algorithmen bis zu innovativen Graph Neural Networks zu erläutern. Dabei richtet sich der Blogbeitrag an Interessierte die ein gewisses Flair für Daten oder Statistik mitbringen und denen der Begriff "Modell" nicht ganz neu ist.

Laut den Vereinten Nationen werden jährlich 2 bis 5 % des globalen BIP – etwa 800 Milliarden bis 2 Billionen US-Dollar – durch Geldwäsche verschleiert. Ein wiederkehrendes Problem in der Geldwäsche-Forschung ist die Verfügbarkeit realer Datensätze. Das AMLworld-Framework bietet hier eine Lösung, indem es synthetische Finanztransaktionen generiert, die reale Szenarien mit hoher Präzision nachbilden, einschliesslich bekannter Geldwäschemuster. Diese vollständig gelabelten Daten ermöglichen eine objektive Bewertung von Algorithmen (Altman et al., 2024). Grundlage des AMLworld-Frameworks ist der synthetische Datensatz [IBM Transactions for Anti Money Laundering (AML)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml), mit dem auch wir in unserem Projekt arbeiteten und den wir im nächsten Kapitel vorstellen. 


# Die Daten
Der Datensatz umfasst 5.078.345 Transaktionen und 11 Variablen. Er beschreibt Transaktionen zwischen Banken und Konten, einschliesslich Zeitstempeln, Beträgen, Währungen, Zahlungsformaten und Labels, die Transaktionen als legal oder Geldwäsche kennzeichnen. 
Die Transaktionen erstrecken sich über 17 Tage ab dem 1. September 2022. Die meisten Daten stammen aus den ersten 10 Tagen, während die restlichen Tage weniger Aktivität zeigen. Auffällig ist das starke Ungleichgewicht: 99,9 % der Transaktionen sind legal, nur 0,1 % gelten als Geldwäsche.  
<img src="assets/eda_tage.png" alt="eda_tage" class="hover-zoom" style="float: left; margin-right: 20px; width: 200px;">

Der Datensatz umfasst 30.470 Banken und 515.080 Konten. Eine kleine Anzahl von Banken und Konten wickelt den Grossteil der Transaktionen ab: Die 10 aktivsten Banken verantworten 18,1 % aller Transaktionen. Bei den Konten gibt es zentrale Akteure, von denen einige über 100.000 Transaktionen ausführen, während viele andere nur ein- oder zweimal aktiv sind. 
<img src="assets/eda_top30banken.png" alt="eda_top30banken" class="hover-zoom" style="float: right; margin-left: 20px; width: 200px;">

Die Transaktionsbeträge, sowohl ein- als auch ausgehend, variieren stark. Meistens handelt es sich um kleine Beträge, doch einige extrem hohe Summen (bis zu 1 Billion USD) verzerren den Durchschnitt. Zur besseren Analyse wurden die Beträge in US-Dollar umgerechnet. 
<img src="assets/eda_beträge.png" alt="eda_beträge" class="hover-zoom" style="float: right; margin-left: 20px; width: 200px;">

US-Dollar und Euro dominieren die Transaktionen, während Währungen wie Bitcoin oder Saudi Riyal selten vorkommen. Interessant ist, dass 98,6 % der Transaktionen in derselben Währung erfolgen, nur 1,4 % beinhalten Währungsumrechnungen. 
Der Datensatz unterscheidet sieben Zahlungsmethoden, darunter Schecks, Kreditkarten, ACH (Automated Clearing House), Bargeld und Bitcoin. Schecks dominieren, während Bitcoin, trotz seiner zunehmenden Nutzung in der Geldwäsche, selten vorkommt.  
<img src="assets/eda_zahlungsformat.png" alt="eda_zahlungsformat" class="hover-zoom" style="float: left; margin-right: 20px; width: 200px;">

Ein auffälliges Merkmal von Geldwäsche ist der hohe Betrag der Transaktionen, die oft am oberen Ende der Skala liegen. Während legale Transaktionen meist kleinere Summen umfassen, bewegen sich illegale häufiger in höheren Bereichen. Dies zeigt, dass grosse Geldbeträge in wenigen Schritten verschoben werden, um die Herkunft zu verschleiern.
<img src="assets/eda_laundering_beträge.png" alt="eda_laundering_beträge" class="hover-zoom" style="float: right; margin-left: 20px; width: 200px;">

Auch die Aktivität einzelner Konten liefert Hinweise auf Geldwäsche. Einige Konten im Datensatz zeigen eine überdurchschnittlich hohe Anzahl an ausgehenden Transaktionen. Solche Konten könnten als "Mule Accounts" fungieren, also Zwischenstationen, über die illegale Gelder fliessen. Die Analyse zeigt, dass diese Konten oft zentrale Knotenpunkte in Transaktionsnetzwerken bilden und deshalb besonders auffallen.
<img src="assets/eda_laundering_accounts.png" alt="eda_laundering_accounts" class="hover-zoom" style="float: left; margin-right: 20px; width: 200px;">

Ein weiteres typisches Muster ist die elektronische Überweisung per ACH. Diese geht oft auf Konten bei anderen Banken. Dabei fällt der Saudi-Riyal besonders auf, neben den häufig genutzten Währungen wie US-Dollar und Euro, unabhängig davon, ob die Transaktion legitim oder betrügerisch ist. 
<div style="display: flex; justify-content: space-between; align-items: center;">
    <img src="assets/eda_laundering_zahlungsformate.png" alt="eda_laundering_zahlungsformate" class="hover-zoom" style="width: 30%; margin: 5px;">
    <img src="assets/eda_laundering_banks.png" alt="eda_laundering_banks" class="hover-zoom" style="width: 30%; margin: 5px;">
    <img src="assets/eda_laundering_währung.png" alt="eda_laundering_währung" class="hover-zoom" style="width: 30%; margin: 5px;">
</div>

Im Kampf gegen Geldwäsche ist es auch entscheidend, typische Muster in Transaktionen zu erkennen, die illegale Aktivitäten verraten. Solche Muster beschreiben spezifische Verhaltensweisen, die darauf abzielen, die Herkunft illegaler Gelder zu verschleiern und sie als legitim erscheinen zu lassen. Im Datensatz sind diese Muster häufig gekennzeichnet, sodass die meisten Transaktion einem bestimmten Muster zugeordnet werden können. Diese Kennzeichnung basiert auf bekannten Geldwäschemechanismen und umfasst Cycles, Scatter-Gather-Strukturen, Chains und andere komplexe Netzwerke. Nachfolgend stellen wir zwei der prominentesten Geldwäsche-Muster aus dem Datensatz vor: Simple Cycles und Scatter-Gather-Strukturen.  

## Simple Cycles 
Das Simple Cycles-Muster beschreibt eine geschlossene Kette von Transaktionen, bei der Gelder innerhalb eines festen Kontenkreises zirkulieren. Diese Strategie zielt darauf ab, die ursprüngliche Herkunft der Gelder durch mehrfache Überweisungen zu verschleiern. 
<img src="assets/cycle.png" alt="cycle" class="hover-zoom" style="float: left; margin-right: 20px; width: 100px;">

Beispiel: 
- Konto A überweist Geld an Konto B.  
- Konto B überweist einen Teil oder den gesamten Betrag an Konto C.  
- Konto C überweist schliesslich das Geld zurück an Konto A.  

Dieses Verhalten zeigt sich oft in der Layering-Phase der Geldwäsche, wenn man Gelder durch verschiedene Konten schleust, um die Spur zu verwischen. Simple Cycles wirken zunächst legitim, doch ihre wiederholte Struktur und der fehlende wirtschaftliche Zweck entlarven sie. In den zuvor vorgestellten Modellen gelten sie als besonders schwer erkennbar, vor allem bei mehr als sechs beteiligten Konten. 

## Scatter-Gather-Strukturen 
Ein weiteres verbreitetes Muster ist die Scatter-Gather-Struktur, die oft in der Integrationsphase der Geldwäsche verwendet wird. Dieses Verhalten besteht aus zwei klar unterscheidbaren Teilen:  

<img src="assets/scatter-gather.png" alt="gather-scatter" class="hover-zoom" style="float: right; margin-left: 20px; width: 100px;">
<img src="assets/gather-scatter.png" alt="gather-scatter" class="hover-zoom" style="float: right; margin-left: 20px; width: 100px;">

Scatter: Gelder werden von einem zentralen Konto auf mehrere Empfängerkonten verteilt.
Gather: Die Gelder fliessen anschliessend von diesen Empfängerkonten zurück auf ein oder mehrere zentrale Konten.

Beispiel: 
- Konto A überweist Gelder an die Konten B, C und D (Scatter).  
- Konten B, C und D leiten diese Gelder zurück an Konto E (Gather).  

Diese Struktur verschleiert die Geldspur durch Streuung und spätere Zusammenführung. Scatter-Gather-Muster sind oft hochgradig organisiert und schwer zu erkennen, da ähnliche Verhaltensweisen auch in legitimen Transaktionsnetzen vorkommen können.  


# Herausforderung und Evaluierung von Modellen 
Zunächst müssen wir verstehen, welche Herausforderungen Machine-Learning-Modelle bewältigen und wie wir ihre Leistung messen. In der Welt des maschinellen Lernens begegnen wir oft Datensätzen mit unausgewogener Klassenverteilung. Das bedeutet, eine Klasse – etwa betrügerische Transaktionen – tritt deutlich seltener auf als die andere, wie legitime Transaktionen. Diese seltene Klasse nennen wir Minority-Class (Minderheitsklasse).

Die Herausforderung bei der Arbeit mit solchen Daten besteht darin, dass herkömmliche Metriken wie die Genauigkeit (Accuracy) oft in die Irre führen. Ein Modell könnte etwa 99 % Genauigkeit erzielen, indem es stets die Mehrheitsklasse (legitime Transaktionen) vorhersagt, dabei jedoch keine betrügerischen Transaktionen erkennt. Hier greift der Minority-Class F1-Score ein. Der F1-Score balanciert Präzision und Recall aus. Diese beiden Masse sind entscheidend, um die Leistung eines Modells bei der Erkennung der Minderheitsklasse zu bewerten. 

Präzision gibt an, wie viele der als "betrügerisch" eingestuften Transaktionen tatsächlich betrügerisch sind. 

$$
\text{Präzision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}
$$

Recall zeigt, wie viele der tatsächlich betrügerischen Transaktionen das Modell erkennt. 

$$
\text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}
$$

Der F1-Score, das harmonische Mittel dieser beiden Werte, berechnet sich so: 

$$
\text{F1} = 2 \cdot \frac{\text{Präzision} \cdot \text{Recall}}{\text{Präzision} + \text{Recall}}
$$  

Ein hoher F1-Score für die Minderheitsklasse zeigt, dass das Modell sowohl präzise als auch sensibel seltene Klassen erkennt und so eine faire, aussagekräftige Bewertung bei unausgewogenen Daten ermöglicht. 

> Ein einfaches Beispiel: Stellen wir uns einen Datensatz mit 50 Transaktionen vor, von denen 2 % betrügerisch sind. Eine
> Transaktion ist betrügerisch (True Positive, wenn korrekt erkannt). Das Modell markiert jedoch fälschlicherweise zwei 
> weitere Transaktionen als betrügerisch (False Positives) und übersieht die betrügerische Transaktion (False Negative).  

