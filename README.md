# ScanBIM Studio

## Repository und Entwicklung

Nach dem Klonen im Projektordner `Start.bat` ausführen (Python 3.12).
Für eigene Python-Experimente zusätzlich Docker Desktop mit Linux-Containern
installieren und `Setup-Experimente.ps1` ausführen. API-Zugänge werden über
die KI-Einstellungen der lokalen Oberfläche eingerichtet.

Entwicklungsabhängigkeiten und Tests:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

Das Repository enthält Quellcode, lokale Webbibliotheken samt deren Lizenz,
Docker-Laufzeit und kleine synthetische Beispiele. `data/`, lokale Zugangsdaten,
die virtuelle Python-Umgebung, erzeugte Beispiel-Läufe und temporäre Inhalte
unter `sandbox/` bleiben lokal. `sandbox/Dockerfile` und `sandbox/worker.py`
gehören zum Quellcode. Eigene Scans und Rekonstruktionsergebnisse separat sichern.

Für den eigenen Projektcode ist derzeit keine Open-Source-Lizenz festgelegt.
Die Lizenz der mitgelieferten Three.js-Dateien steht in
`web/vendor/THREE-LICENSE.txt`.

Lokales experimentelles Scan-to-BIM-Labor, Version 0.3. **Eigene parametrische Geometrie ist
die Basis für das Live-IFC-Arbeitsmodell und den Export.** Der visuelle LLM-Agent darf
seine Verfahren frei wählen und eigene NumPy/SciPy-Programme entwickeln.

## Autonomer Rekonstruktionsworkflow

Der visuelle Agent entwickelt und aktualisiert seinen eigenen Bauteilplan. Er untersucht
jeweils einen bearbeitbaren Bereich, misst die Geometrie, baut semantische IFC-Bauteile
und verbessert sie anhand numerischer und visueller Vergleiche mit der Punktwolke.
Die Auswahl der Verfahren, Werkzeuge und Bauteilreihenfolge bleibt beim Modell.
Eigene Python-Werkzeuge laufen synchron in der isolierten Versuchsumgebung: Erst nach
Ergebnis oder Fehlermeldung entscheidet der Agent über den nächsten Schritt.

Im aktuellen Modellkontext bleiben Plan, Bauteilübersicht, Modellrevision und die letzten
lokalen Fit-Ergebnisse sichtbar. Rückmeldungen bei längerer Analyse ohne Modelländerung
regen den Agenten zum Bauen beziehungsweise Vergleichen an; sie erzwingen keinen Algorithmus.
Identische wiederholte Aktionen erhalten einen Hinweis zur Änderung des Vorgehens.
Nach einer Modelländerung sind dieselben Prüfaufrufe erneut möglich. Reines Nachlesen
von Befunden beendet den Lauf nicht vorzeitig. Stopp, Schrittbudget und Fehlergrenzen
bleiben wirksam; ein Teilabschluss benennt offene Arbeiten und exportiert vorhandene Geometrie.

## Vollstaendiges Werkzeugverzeichnis

Der Agent erhaelt alle Werkzeugbeschreibungen und Argumente vollstaendig. Die Referenz
steht in [docs/TOOLS.md](docs/TOOLS.md). Lokale RANSAC-Ebenensuche, DBSCAN-Clustering,
PCA und Normalenanalyse stehen direkt bereit. Ebenen und Cluster liefern gespeicherte
Punktauswahlen (`selection_id`), die sich gezielt weiter analysieren und in eigenen
Experimenten fuer die Bauteilgeometrie verwenden lassen. Selbst geschriebener Python-Code
wird weiterhin lokal im isolierten Docker-Versuchsraum ausgefuehrt.

## Sparsamer Modellkontext

Der visuelle Agent sendet nur einen aktuellen Szenenstand, bis zu drei vollständige
Werkzeugrunden und die neueste Ansicht. Alte Szenenstände und Bilder werden nicht
erneut angehängt. Ansichten werden standardmäßig mit hoher Detailstufe übertragen, damit dünne
Bauteile erkennbar bleiben. Für einfache Übersichten kann das Modell mit
`image_detail: "low"` eine auf 512 Pixel begrenzte Ansicht wählen. Originalbilder
und numerische Messwerte bleiben erhalten.

Werkzeugergebnisse über 5.000 Zeichen und Szenenstände über 6.500 Zeichen werden
vollständig unter `agent-results` gespeichert und als kompakte Vorschau übergeben.
`read_agent_result` liest gezielt über JSON Pointer und Seiten; ohne Ergebnis-ID
listet es das Archiv. Eigener Code und komplette Geometrien bleiben so abrufbar.
Berechnungen sollen lokal kompakte Kennwerte liefern, Geometrien werden direkt aus
gespeicherten Experimenten übernommen. Das ausgewählte Modell entscheidet weiter
selbst über seine Verfahren.

Die Oberfläche zeigt die vom Anbieter gemeldeten Eingabe-, Ausgabe- und Cache-Tokens.
Einzelwerte stehen im Protokoll unter `model_usage`. Das ist keine vollständige
Abrechnung: fehlgeschlagene Anfragen ohne Verbrauchsangabe werden nicht erfasst.
Einsparung und Rekonstruktionsqualität hängen vom konkreten Lauf ab.

Gezielter Test ohne API-Aufrufe: `.venv\Scripts\python.exe -m unittest discover -s tests -p test_token_context.py -v`.

## Start unter Windows

1. Python **3.12** mit Python Launcher installieren. Internet wird einmal für die
   Paketinstallation benötigt. Python 3.14 ist wegen der IFC-Laufzeit nicht das Ziel.
2. `Start.bat` doppelklicken. Beim ersten Start wird eine lokale `.venv` eingerichtet.
3. [Studio öffnen](http://127.0.0.1:8765). Das Startfenster geöffnet lassen.
4. Zum lokalen Prüfen: **Referenzraum öffnen → Referenzdemo wählen → Untersuchung starten**. Der Demo-Agent untersucht
   die Daten, fordert Kamerabilder an, rekonstruiert Flächen/Öffnungshypothesen,
   validiert und erzeugt IFC4. Über **IFC exportieren** herunterladen.

Alternativ im Terminal:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:SCANBIM_PROVIDER = 'openai'
.\.venv\Scripts\python.exe -m uvicorn scanbim.app:app --host 127.0.0.1 --port 8765
```

Die Three.js-Bibliothek ist lokal enthalten; kein CDN und kein Node-Build werden
beim Betrieb benötigt. Der Dienst bindet nur an Loopback. Nicht unverändert ins
Internet stellen. Importierte Punktdaten und Zustände liegen in `data/` und
bleiben nach dem Neustart erhalten. `SCANBIM_DATA_DIR` kann den Speicherort ändern.

## Live-IFC und autonome Nacharbeit

Im dritten Ansichtsmodus **BIM-Körper** startet **BIM-Modell erstellen** die
volumetrische Rekonstruktion. Das Studio zeigt die Gebäudestruktur oder die
Anlagenteile und enthält im BIM-Download dieselben Körper und Geländeoberflächen.
Offene Bauteilmessflächen erscheinen weiterhin im Arbeitsstand und im Verlauf.
Ein Teilstand kann als `bim.ifc` exportiert werden.

Zwei zugehörige Messflächen können einen Körper mit variabler Dicke begrenzen.
Die Rekonstruktion erzeugt dann einen geschlossenen IFC-BRep. Eine einzelne
nahezu ebene Sichtfläche lässt sich als Profilkörper extrudieren, wenn Richtung
und Tiefe begründet sind. Der Agent kann im Scan nach möglichen Rückseiten
suchen. Annahmen, ergänzte Gegenflächen und Messquellen stehen am Bauteil und
in dessen IFC-Nachweis. Bei Brücken lassen sich Körper etwa unter Überbau,
Widerlager und Pfeiler als `IfcBridgePart` gliedern.

- Rechts zeigt das Studio die von IfcOpenShell ausgewerteten IFC-Körper. In **IFC-Modell**
  stehen Bauteilklassen, Volumenkörper und Koordinationsbefunde. Rote Markierungen zeigen
  Schnittvolumina. Am ausgewählten Bauteil lassen sich stützende Scanpunkte einblenden.
- Der Verlauf öffnet die damals gültige Geometrie und die Versuchsebenen. **Versuche**
  zeigt Quellcode und Ausgaben. Der IFC-Download bezieht sich auf den aktuellen Live-Stand.
- Der visuelle Agent kann aus eigenen Experimenten offene Oberflächen und geschlossene
  Netzkörper übernehmen, Gelände aus regionalen Punkten ableiten, Verschnitte und
  Öffnungen berechnen und Klassifikation, Typ, Materialschichten und Beziehungen ergänzen.
  Projekte können IFC4 oder IFC 4.3 verwenden. Ein bekannter Vermessungsbezug kann
  als Koordinatentransformation hinterlegt werden.
- Ein vollständiger Abschluss verlangt Punkt- und IFC-Prüfung. Teilstände werden
  ebenfalls exportiert und im Protokoll als solche geführt. Die Verfahrensbibliothek
  speichert Versionen über mehrere Szenen hinweg. Zur automatischen Auswahl muss eine
  Netzvariante an mindestens zwei verschiedenen Quellscans die Prüfung bestehen.
- Offene Oberflächen und angenommene Bauteile bleiben erkennbar. Die Geländefunktion
  arbeitet mit einer regionalen Rasterhypothese. Starke Bruchkanten und komplexe
  Verdeckungen können eigene Verfahren und weitere Messdaten erfordern.

## Eigene Daten und Bedienung

- **Punktwolke öffnen → Lokal öffnen**: PLY (ASCII/little-/big-endian binary),
  PCD (ASCII/binary) und XYZ. Große Dateien direkt von der Festplatte wählen.
  Der Import arbeitet blockweise mit Fortschritt und Abbruch. Die alten Grenzen
  von 64 MiB und einer Million Punkten gelten hier nicht mehr. PCD `binary_compressed` wird
  verständlich abgewiesen. E57 und LAS/LAZ sind spätere Importadapter.
- Einheit vor dem Import wählen: m, cm oder mm. Koordinaten bleiben erhalten;
  die interne Einheit ist Meter, Z zeigt nach oben. Andere Hochachsen vorab drehen.
- Beide Kameras sind synchron. Linke Maustaste dreht, rechte verschiebt, Mausrad
  zoomt. Orthografische Ansichten, Grundriss, Schnitt und Überlagerung stehen bereit.
- **Ansicht** erzeugt ein echtes Punktbild mit Kamera-/Projektionsmetadaten.
  Bei aktivem Z-Schnitt wird eine 12 cm starke horizontale Scheibe aufgenommen.
  Frei orientierte Schnitte sind über `request_view` möglich.
- **Messen**: zwei Scanpunkte wählen; die Strecke wird aus Quellpunktkoordinaten
  berechnet. Die Anzeige ist bei großen Daten auf 200.000 Punkte begrenzt;
  Standardanalysen nutzen bis zu 200.000 echte Punkte der gewählten Region und
  kennzeichnen die Stichprobe. Eigene Verfahren können mit `iter_point_blocks()`
  alle Daten der gewählten Region in begrenzten Blöcken lesen.
- Bauteil anklicken: Maße, Messgrundlage, angenommene Dicke, Öffnungen und
  Qualitätswerte einsehen. Manuelle Parameteränderungen sperren das Element
  gegen Agentenänderungen. Sperre lässt sich im Inspektor wieder aufheben.
- **Abweichungen** färbt Scanpunkte nach Abstand zu den endlichen Modellflächen.
  Fehlt ein Scanpunkt in der begrenzten Validierungsstichprobe, behält er seine
  Quellfarbe. RMS und Toleranzanteil beziehen sich auf die bezeichnete Stichprobe.
- **Verlauf** spielt gespeicherte Modellzustände ab. **Live** kehrt zum aktuellen
  Zustand zurück. Journal und internes Modell sind herunterladbar.

## Freier visueller LLM-Agent

Der Demo-Agent ist ein festgelegter geometrischer Integrationstest, **kein LLM**.
Für den offenen visuellen Agenten oben **KI-Einstellungen** öffnen. Dort gibt es
ein verdecktes Feld für den API-Schlüssel und ein Feld für die Modell-ID.
**Verfügbare Modelle laden** fragt die für diesen Schlüssel sichtbaren IDs ab.
Über **Anbieter** zwischen OpenAI, Claude / Anthropic und lokalem Ollama wechseln.
Für OpenAI und Claude werden Schlüssel und Modell-ID getrennt gespeichert.
**Text testen** prüft eine kurze Antwort. **Bild testen** lässt die zufällig
gewählte Farbe eines kleinen Testbilds erkennen. **Werkzeug testen** prüft einen
strukturierten Aufruf. **Workflow testen** kombiniert Bildinput, Werkzeugaufruf
und eine zweite Modellantwort nach dem Werkzeugergebnis.
Die Tests verwenden synthetische Eingaben und können API-Kosten verursachen.
Mit **Lokal speichern** werden die Eingaben sofort für den Agenten wirksam.

Der Speicherort bleibt `%LOCALAPPDATA%\ScanBIM-Studio\settings.toml`.
**API-Einstellungen.bat** kann diese Datei weiterhin im Texteditor öffnen. Im Projekt
steht nur eine leere Vorlage (`settings.example.toml`). Die tatsächliche Datei
liegt außerhalb des Projektordners und wird weder in Git noch in das
Quellpaket, IFC, Journal oder Docker übernommen.

```toml
[studio]
provider = "openai"

[openai]
api_key = ""
model = ""

[claude]
api_key = ""
model = ""
```

Nach dem Speichern wird das Schlüsselfeld geleert. Ein leer gelassenes Feld
behält den vorhandenen Schlüssel; zum Ersetzen einen neuen eintragen. Der Server
gibt den Schlüssel nie an die Oberfläche zurück, und JavaScript speichert ihn
weder in localStorage noch in sessionStorage. Beim Eingeben wird er nur an den
lokalen Server übermittelt. Ein Serverneustart ist nicht erforderlich.
Die bisherigen Umgebungsvariablen bleiben als Rückfall nutzbar; explizit lokal
gespeicherte Werte haben Vorrang. Die Dateilesung akzeptiert auch UTF-8-BOM und
UTF-16, damit Notepad-Speicherformate nicht zu unklaren Konfigurationsfehlern führen.
Bei Modellanfragen wird der Schlüssel ausschließlich zur Anmeldung an
beim ausgewählten Anbieter gesendet. OpenAI verwendet
`https://api.openai.com`, Claude `https://api.anthropic.com`. Er wird nicht an das Sprachmodell als Inhalt
oder Experimentprogramme weitergereicht. Eine normale lokale
Textdatei ist kein Schutz gegen jemanden, der Zugriff auf das Benutzerkonto hat.

Dann **Visueller LLM · freie Experimente** auswählen und starten. Die Modell-ID
ist absichtlich konfigurierbar; es wird keine Modellverfügbarkeit vorausgesetzt.
Schlüssel werden nur im Serverprozess verarbeitet. Dieser Modus überträgt Ansichten,
Werkzeugergebnisse, Geometrie und Szenengedächtnis an die ausgewählte Modell-API und verursacht
API-Kosten. Der Start erfolgt ausdrücklich durch die Moduswahl im Studio.
Maximal 35 Aktionen je UI-Start, über die API höchstens 60. Wiederholungen,
Fehlerfolgen und stagnierende Aktionen führen zum Abbruch. Stop/Pause wirken
zwischen Aktionen; eine laufende Modellanfrage hat ein begrenztes Zeitlimit.

Der Agent erhält weder einen vorgegebenen Rekonstruktionsplan noch eine feste
Werkzeugreihenfolge. Er kann z. B. erst Querschnitte prüfen, eine eigene robuste
Regression programmieren und daraus eine nicht ebene Dachfläche erzeugen.
`run_experiment` ist ein allgemeiner Versuchskanal, kein Dach-Spezialwerkzeug.

Offizielle Schnittstellenquellen:
[Responses Function Calling](https://developers.openai.com/api/docs/guides/function-calling),
[Bildeingaben](https://developers.openai.com/api/docs/guides/images-vision).

### Visueller Start und Werkzeuggedächtnis

Jeder visuelle Run erzeugt zuerst eine echte Übersicht der gesamten Punktwolke.
Das Bild mit Kamerametadaten liegt bereits in der ersten Modellanfrage.
Der Agent soll seine Beobachtung dokumentieren, geeignete frühere Verfahren
suchen und seine Analysewerkzeuge mit `define_tool` entwickeln. Über
`call_custom_tool` testet er sie im Docker-Versuchsraum und korrigiert sie
anhand von Messwerten und Fehlermeldungen.

Eigene Werkzeuge und freie Experimente werden automatisch szenenübergreifend
unter `data/.scanbim-tools/` gespeichert. Das Gedächtnis bewahrt Codeversionen,
Beschreibungen, Parameter, Quellscan und Ergebnisse der Versuche. Auch
fehlgeschlagene Versuche bleiben dokumentiert. `list_tool_memory` durchsucht
das Gedächtnis; `recall_tool` lädt eine gewählte Version in die aktuelle Szene.
Am Beginn eines Runs erhält das Modell einen kompakten Katalog. Nach dem
Zurücksetzen einer Szene oder einem Serverneustart bleibt das Gedächtnis erhalten.
Bestehende lokale Werkzeuge werden beim nächsten Aufruf ebenfalls gespeichert.

Bei erfolgreichen kleinen Netzen bis 5000 Flächen werden zusätzliche
geometrische Prüfwerte bei 30 mm Toleranz aus bis zu 12000 Punkten gespeichert.
Für andere Ergebnisse wird der Ausführungserfolg dokumentiert. Für den
Vergleich verbesserter Netzverfahren gibt es weiterhin `evaluate_method`
mit Baseline und Prüfungen an unabhängigen Quellscans. Verbesserungen werden
anhand der Ergebnisse beurteilt. Die Modellgewichte werden dabei nicht trainiert.

### Ein Modell pro Run

Der ausgewählte Anbieter und die Modell-ID gelten für den gesamten Run.
Bildinterpretation, Verfahrensentwicklung, Klassifikation und Nacharbeit
werden vom selben Modell gesteuert. Die Werkzeuge berechnen die Geometrie
lokal und geben ihre Ergebnisse an dieses Modell zurück. Die Auswahl wird
beim Start des visuellen Ablaufs eingelesen. Änderungen in den KI-Einstellungen
gelten für den nächsten Run.

### Bauteile genauer an die Punktwolke anpassen

Mit `inspect_fit` erhält der Agent für ein Bauteil RMS, Median und P95 der
Abstände, den Punktanteil innerhalb der Toleranz, die Flächenstützung und
Koordinaten der größten Abweichungen. Der Bericht wird gespeichert.
Nach einer Korrektur vergleicht `baseline_id` gegen denselben Quellpunktumfang,
dieselbe Region und dieselbe Toleranz. Die Regionenprüfung kennzeichnet,
ob die gesamte Ausdehnung des Bauteils eingeschlossen ist.

Der Agent soll daraus gezielte Verfahren zur Anpassung von Lage, Neigung,
Abmessungen und Profil entwickeln. Bei planaren Elementen können Ursprung
und orthonormaler Rahmen mit Versuchsnachweis revidiert werden. Netzgeometrie
wird über `create_mesh` unter derselben Element-ID überarbeitet.
Der Agent entscheidet, welche Diagnose- und Anpassungsverfahren er verwendet.
Die lokalen Fit-Berichte sind optional. Die Gesamtprüfung verwendet die globale
Validierung und die vom Agenten gewählten Akzeptanzkriterien. Die lokalen Kennzahlen hängen
von der Prüfregion, Nachbarbauteilen und Verdeckungen ab.

## Zeitraffer

Neue Agentenläufe speichern automatisch PNG-Bilder vom Start, von
Modelländerungen und vom Endstand. Jedes Bild zeigt die Punktwolke und
eine vereinfachte Rekonstruktionsansicht aus derselben festen Perspektive.
Die Aufnahme läuft auf dem Server und funktioniert bei geschlossenem Browser.
Unter **Zeitraffer** neben **Protokoll** einen Lauf auswählen und
**GIF speichern** wählen. Die Animation überspringt Wartezeiten und hält den
Endstand länger sichtbar.

Die Bilder und das exportierte `zeitraffer.gif` liegen pro Lauf in
`data/<Szenen-ID>/timelapse-<Lauf-ID>/` und bleiben auch beim Zurücksetzen
der Rekonstruktion erhalten. Pro Lauf werden bis zu 240 Bilder gespeichert.
Längere Aufnahmen werden für das GIF auf höchstens 120 Bilder verdichtet.

## Eigene Experimente

1. Docker Desktop installieren und **Linux-Container** verwenden. `Start.bat`
   startet Docker Desktop bei Bedarf und startet es erneut, wenn es bei
   laufendem Studio beendet wurde. Ein Experiment wartet beim Start auf eine
   vorübergehend nicht erreichbare Docker-Engine.
2. `Setup-Experimente.ps1` ausführen. Der erste Build lädt Python, NumPy und SciPy.
3. Im Studio **Experimente → Neuer Versuch** öffnen. Optional vorher eine Region
   auswählen. Eigene Algorithmen mit `points`, `colors`, `np`, `scipy` schreiben.
4. `result` auf ein JSON-kompatibles Objekt setzen. `result.overlays` kann
   Punkte, Linien und Netze in der 3D-Szene darstellen. Fehler werden gespeichert;
   Quellcode lässt sich korrigieren und erneut ausführen.
5. `examples/variable-roof.ply` laden. Das beigelegte Dachverfahren lässt sich
   mit **Dachverfahren laden** öffnen.
   Es schätzt First-/Traufverläufe und veränderliche Dachneigungen pro Schnitt.
   Es setzt ein überwiegend sattelförmiges Dach voraus und dokumentiert diese
   Annahme. Es ist kein allgemeiner Dachdetektor.
6. Ein Ergebnis mit `vertices` und `faces` lässt sich als freies Netzbauteil
   übernehmen. Der Agent hat dieselbe Möglichkeit über `create_mesh`.

Jeder Versuch läuft in einem neuen Container: kein Netzwerk, kein beschreibbarer
Host-Mount, 1 CPU, 512 MB RAM, 64 Prozesse, maximal 30 Sekunden, begrenzte Ausgabe.
Es gibt **keinen Rückfall auf Host-Ausführung**. Ohne Docker funktionieren Viewer,
Standardgeometrie, Demo, Validierung und IFC weiter. Details in
[Architektur](docs/ARCHITEKTUR.md) und [Tool-Vertrag](docs/TOOLS.md).

## Reproduzierbarer Durchlauf und Tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m scanbim.cli --output data\demo-run
.\.venv\Scripts\python.exe -m pytest -q
# Optional: echte Containergrenzen und Versuchsausführung prüfen
$env:SCANBIM_TEST_DOCKER = '1'
.\.venv\Scripts\python.exe -m pytest tests/test_sandbox.py -q
```

Die CLI erzeugt eine PLY-Beispielwolke, Bilder, Journal, IFC und `report.json`.
Der Seed ist fest. `examples/reference-room.ply` kann direkt importiert werden.
Die Tests prüfen Importvarianten, Einheiten, Geometrie, Unsicherheiten,
Öffnungsausschnitte, IFC-Struktur/-Geometrie, Persistenz, manuelle Sperren,
API-Grenzen und die angeforderte Bildübergabe im simulierten VLM-Regelkreis.
Der geprüfte Raumexport liegt unter `examples/reference-room.ifc`, der freie
Dachexport unter `examples/variable-roof.ifc`. Die vollständigen Nachweise liegen
in `examples/reference-run/` und `examples/roof-run/`.

## Projektstruktur und Grenzen

Siehe [Architektur und Entscheidungen](docs/ARCHITEKTUR.md),
[HTTP-/Werkzeugschnittstelle](docs/TOOLS.md) und
[Prüfergebnisse](docs/PRUEFBERICHT.md). Der getestete Referenzfall ist ein einzelner Raum mit planaren Begrenzungen und rechteckigen Punktlücken. Der autonome Agent ist nicht auf diesen Gebäudetyp beschränkt; seine Zuverlässigkeit für beliebige reale Bauwerke ist noch nicht nachgewiesen.
Mehrraumtopologie, sichere Tür-/Fenstersemantik, allgemeine Dachrekonstruktion,
Tragwerksverständnis und Sichtbarkeitsanalyse sind weitere Entwicklungsschritte.
Es werden keine unbeobachteten Tür-/Fensterfüllungen erfunden. Im IFC stehen die
Lücken als `IfcOpeningElement`, mit Hypothesenstatus und Rasterauflösung.

IfcOpenShell verwendet LGPL-3.0-or-later, Three.js MIT; die Three.js-Lizenz liegt
unter `web/vendor/THREE-LICENSE.txt`. Paketlizenzen bleiben bei ihren Projekten.

## Große Punktwolken (5–20 GB)

**Punktwolke öffnen → Durchsuchen → Lokal öffnen** vermeidet eine Übertragung
durch den Browser und eine zusätzliche Kopie der Originaldatei. Alternativ den
vollständigen Dateipfad eintragen. Die Quelldatei wird nicht verändert.

Beim ersten Import entstehen `points.npy` (Float64, Meter), `colors.npy`, ein
Blockindex und eine Vorschau im Szenencache. Zusätzlicher Plattenbedarf: etwa
**27 Byte pro Punkt**. Ein 20-GB-Binär-PLY mit 15 Byte pro Punkt kann deshalb
ungefähr 36 GB Cache benötigen. Der Import prüft den freien Speicher und kann
abgebrochen werden. Abgebrochene Arbeitsdateien werden entfernt. Es läuft
höchstens eine Konvertierung gleichzeitig. Das konfigurierbare Dateilimit liegt
standardmäßig bei 64 GiB (`SCANBIM_MAX_FILE_GB`).

Der Browser erhält höchstens 200.000 Punkte mit stabilen Quellindizes. Alle
Originalpunkte bleiben im Festplattencache zugänglich. Große Weltkoordinaten
werden für die Anzeige relativ zum Szenenzentrum dargestellt; Messungen und
Exporte behalten die Quellkoordinaten.

Regionale Analysen lesen über Blockgrenzen, ohne einen Filter über die komplette
Punktwolke im RAM anzulegen. Standardwerkzeuge arbeiten mit gekennzeichneten
Stichproben. Eigene Algorithmen können mit `iter_point_blocks()` die vollständigen
Punkte einer Region lesen; siehe `examples/stream_experiment.py`. Bei sehr großen
Regionen gelten weiterhin Zeit- und Speichergrenzen des Experimentcontainers.
Der Blockindex ist noch kein räumlicher Octree; bei unsortierten Scans kann eine
Regionsabfrage viele Blöcke lesen müssen.

Getestet: synthetische **5 GiB / 357.913.941 Punkte**, vollständig importiert und
wieder geöffnet; zusätzlich ein 81-MiB-ASCII-PLY und PCD mit mehr als einer
Million Punkten. Ein realer 20-GB-Scan wurde noch nicht getestet.

## Projektpaket ohne Zugangsdaten

```powershell
.\.venv\Scripts\python.exe -m scanbim.package --output ..\ScanBIM-Studio.zip
```

Der Packager verwendet eine Dateiliste für Quellcode und Beispieldaten. Lokale
Einstellungen, `.env*`, `*.local.toml`, Laufzeitdaten und private Ordner werden
ausgeschlossen; bei erkennbaren API-Schlüsseln im übrigen Quellcode bricht er ab.


## Version 0.3: offene Rekonstruktion und sichtbarer Prozess

Der visuelle LLM ist die vorausgewählte Betriebsart. Er setzt weder einen Raum
noch eine bestimmte Bauform voraus. Ebene, Profil, gekrümmte Schale, Brücke oder
unregelmäßige Oberfläche können mit selbst gewählten Verfahren untersucht werden.
`define_tool` speichert eigene Python-Verfahren mit Parameterbeschreibung;
`call_custom_tool` führt sie wiederholt im bestehenden Docker-Versuchsraum aus.
Sie bleiben mit Code und Revision nach Neustarts verfügbar. Die optionalen
Standardwerkzeuge sind keine vorgeschriebene Abfolge.

Öffentliche Entscheidungen einschließlich Begründung, Nachweisen, Konfidenz und
Revision erscheinen im Gedächtnis. Neue Importe kennzeichnen echte RGB-Quellfarben
oder reine Anzeigefarben. Eigene Farb-/Raumanalysen können farbige Overlays und
Materialhypothesen erzeugen. Materialhypothesen bleiben unbestätigt und werden als
solche nach IFC exportiert. Alte Caches ohne Farbherkunft gelten als unbekannt;
für RGB-Materialhypothesen neu importieren.

Der Prozessbereich zeigt aktuelle Tätigkeit und Werkzeug schon während langer
Berechnungen. Die Kamera fährt weich zur angeforderten Ansicht oder zum
Arbeitsbereich. Manuelles Drehen pausiert die Kameraführung; das Häkchen oder
das Vorschaubild holt sie zurück. Das Vorschaubild ist die tatsächlich erzeugte
Scanansicht; es ist von der frei bedienbaren 3D-Darstellung getrennt. Im VLM-Modus
geht diese angeforderte Ansicht auch an das Modell. Der Verlauf stellt gespeicherte
Ansichten wieder her. Kurze Aktionen können zwischen zwei Live-Aktualisierungen
abgeschlossen sein; jede bleibt im Verlauf enthalten.

`review_model` prüft das Gesamtmodell gegen die aktuelle globale Validierung,
Messnachweise, identische doppelte Geometrie und entartete Dreiecke. Der Agent
wählt passende RMS-/Punktabdeckungskriterien und begründet sie. Er muss zusätzlich
räumliche Anschlüsse und Widersprüche beurteilen. `finish(status="complete")`
verlangt eine bestandene Prüfung und erzeugt IFC. Ein erschöpftes Budget oder
offene Befunde ergeben einen **Zwischenstand**, keine bestätigte Fertigstellung.
Diese Prüfung ersetzt keine Tragwerks- oder vollständige Topologieprüfung.

**Run zurücksetzen** entfernt nach angezeigtem Löschumfang und Bestätigung alle
abgeleiteten Ergebnisse der aktuellen Szene, einschließlich manueller Modelländerungen,
eigener Werkzeuge, Verlauf, Ansichten und IFC. Die Punktwolke samt Quellcache,
andere Szenen und der KI-Zugang bleiben erhalten. Während eines aktiven oder
pausierten Runs ist Zurücksetzen gesperrt; erst stoppen und das Laufende abwarten.
Die Revision wird bei Bestätigung geprüft, damit ein veralteter Dialog keine
zwischenzeitlichen Änderungen löscht. Nach erfolgreichem Löschen gibt es keine
Wiederherstellung im Studio.
