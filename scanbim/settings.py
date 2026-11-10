"""Private editable configuration lives outside the project and its exports."""
from __future__ import annotations
from dataclasses import dataclass,field
import os
import json
from pathlib import Path
import re
import tempfile
import threading
import tomllib
import httpx
from . import ollama, claude

_lock=threading.RLock()

TEMPLATE='''# Nur lokale Einstellungen für ScanBIM Studio.
# Diese Datei liegt AUSSERHALB des Projektordners und seiner ZIP-/IFC-Exporte.
# Bequem über KI-Einstellungen in der Studio-Oberfläche bearbeiten.
# Nicht in Quellcode, Chat oder Experimentprogramme übernehmen.

[openai]
api_key = ""
model = ""

# api_key: Ihren persönlichen API-Schlüssel zwischen die Anführungszeichen setzen.
# model: Modell-ID mit Bildinput und Responses-Function-Calling eintragen.
# Der Schlüssel wird ausschließlich zur Anmeldung an https://api.openai.com benutzt.
'''

def settings_path():
    custom=os.getenv("SCANBIM_SETTINGS_FILE")
    if custom:return Path(custom).expanduser()
    base=Path(os.getenv("LOCALAPPDATA",str(Path.home()/"AppData"/"Local"))) if os.name=="nt" else Path.home()/".config"
    return base/"ScanBIM-Studio"/"settings.toml"

def active_provider():
    try:
        selected = _read_values(section="studio").get("provider")
        if selected in ("openai", "claude", "ollama"):
            return selected
    except ValueError:
        pass
    selected = os.getenv("SCANBIM_PROVIDER", "").lower()
    return selected if selected in ("openai", "claude") else "ollama"


def resolve_provider(provider=None):
    provider = active_provider() if provider is None else provider
    if provider not in ("openai", "claude", "ollama"):
        raise ValueError("Anbieter OpenAI, Claude oder Ollama auswählen.")
    return provider


def local_settings_path():
    return settings_path().with_name("ollama.toml")


def local_credentials(model):
    if not ollama.is_local_model(model):
        raise ValueError("Bitte eine gültige lokale Ollama-Modell-ID eintragen.")
    return Credentials("", model, "ollama")


def initialize_settings():
    local = active_provider() == "ollama"
    path=local_settings_path() if local else settings_path()
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        with path.open("x",encoding="utf-8") as f:
            f.write(f'[ollama]\nmodel = "{ollama.DEFAULT_MODEL}"\n' if local else TEMPLATE)
        if os.name!="nt":path.chmod(0o600)
    return path

@dataclass(frozen=True)
class Credentials:
    api_key:str=field(repr=False)
    model:str=""
    provider:str="openai"
    @property
    def configured(self):return bool(self.model and (self.provider == "ollama" or self.api_key))

def _read_values(path=None, section="openai"):
    values={};path=path or settings_path()
    if path.is_file():
        try:
            with path.open("rb") as f:raw=f.read(32769)
            if len(raw)>32768:raise ValueError("Zu große Einstellungsdatei.")
            encoding="utf-16" if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else "utf-8-sig"
            values=tomllib.loads(raw.decode(encoding)).get(section,{})
        except (OSError,UnicodeError,tomllib.TOMLDecodeError):
            raise ValueError(f"Einstellungsdatei ungültig. Anführungszeichen und [{section}] prüfen.") from None
        if not isinstance(values,dict):raise ValueError(f"Einstellungen benötigen einen [{section}]-Abschnitt.")
    return values

def _validate(key,model):
    if not isinstance(key,str) or not isinstance(model,str):raise ValueError("Schlüssel und Modell müssen Text in Anführungszeichen sein.")
    key,model=key.strip(),model.strip()
    if len(key)>4096 or any(ord(c)<33 or ord(c)>126 for c in key):
        raise ValueError("API-Schlüssel ohne Leerzeichen, Zeilenumbrüche oder Anführungszeichen eintragen.")
    if any(c in key for c in ('"',"'")):
        raise ValueError("API-Schlüssel ohne umgebende Anführungszeichen eintragen.")
    if model and (len(model)>200 or model.startswith("sk-") or not re.fullmatch(r"[A-Za-z0-9._:/-]+",model)):
        raise ValueError("Im Feld model bitte nur die Modell-ID eintragen; den Schlüssel ausschließlich unter api_key.")
    return Credentials(key,model)

def load_credentials(provider=None):
    with _lock:
        provider=resolve_provider(provider)
        if provider == "ollama":
            values = _read_values(section="ollama") or _read_values(local_settings_path(), "ollama")
            return local_credentials(values.get("model", ollama.DEFAULT_MODEL))
        values=_read_values(section=provider)
        # Values saved in the UI take effect immediately; environment variables
        # remain a fallback for installations without local settings.
        prefix="ANTHROPIC" if provider=="claude" else "OPENAI"
        credentials=_validate(values.get("api_key") or os.getenv(prefix+"_API_KEY", ""),
                              values.get("model") or os.getenv(prefix+"_MODEL", ""))
        return Credentials(credentials.api_key,credentials.model,provider)

def settings_status(provider=None):
    provider=resolve_provider(provider)
    try:
        credentials=load_credentials(provider)
        return {"has_api_key":bool(credentials.api_key),"model":credentials.model,
                "configured":credentials.configured,"provider":credentials.provider,"error":None}
    except ValueError as error:
        return {"has_api_key":False,"model":"","configured":False,"provider":provider,"error":str(error)}

def resolve_draft(api_key,model,provider=None):
    provider=resolve_provider(provider)
    if not isinstance(api_key,str) or not isinstance(model,str):
        raise ValueError("API-Schlüssel und Modell müssen Text sein.")
    key=api_key.strip()
    if not key:
        try:
            values=_read_values(section=provider)
            key=values.get("api_key") or os.getenv(("ANTHROPIC" if provider=="claude" else "OPENAI")+"_API_KEY", "")
        except ValueError:
            raise ValueError("Bitte den API-Schlüssel erneut eingeben; die bisherige Datei konnte nicht gelesen werden.") from None
    credentials=_validate(key,model)
    if not credentials.api_key:raise ValueError("Bitte einen API-Schlüssel eintragen.")
    return Credentials(credentials.api_key,credentials.model,provider)

def _replace_section(text,section,values):
    replacement="["+section+"]\n"+"".join(key+" = "+json.dumps(value,ensure_ascii=False)+"\n"
                                          for key,value in values.items())
    pattern=r"(?ms)^\["+re.escape(section)+r"\][^\n]*(?:\n|\Z).*?(?=^\[|\Z)"
    if re.search(pattern,text):
        return re.sub(pattern,lambda match: replacement+"\n",text,count=1)
    return text.rstrip()+"\n\n"+replacement


def save_settings(api_key,model,provider=None):
    with _lock:
        provider=resolve_provider(provider)
        if provider == "ollama":
            credentials = local_credentials(model.strip())
        else:
            credentials=resolve_draft(api_key,model,provider)
            if not credentials.model:raise ValueError("Bitte eine Modell-ID eintragen.")
        path=settings_path()
        try:
            _read_values(path)
            raw=path.read_bytes() if path.is_file() else b""
            payload=raw.decode("utf-16" if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else "utf-8-sig")
        except (ValueError,OSError,UnicodeError):
            payload=""
        values={"model":credentials.model}
        if provider!="ollama":values["api_key"]=credentials.api_key
        payload=_replace_section(payload,provider,values)
        payload=_replace_section(payload,"studio",{"provider":provider})
        temporary=None
        try:
            path.parent.mkdir(parents=True,exist_ok=True)
            descriptor,temporary=tempfile.mkstemp(prefix=".settings-",suffix=".tmp",dir=path.parent)
            with os.fdopen(descriptor,"w",encoding="utf-8",newline="\n") as f:f.write(payload)
            if os.name!="nt":Path(temporary).chmod(0o600)
            Path(temporary).replace(path)
        except OSError:
            raise ValueError("Lokale Einstellungen konnten nicht gespeichert werden. Schreibzugriff auf den lokalen Einstellungsordner prüfen.") from None
        finally:
            if temporary:Path(temporary).unlink(missing_ok=True)
        return settings_status()

def _provider_error(status):
    return {401:"OpenAI akzeptiert diesen API-Schlüssel nicht. Schlüssel prüfen oder neu erstellen.",
            403:"Dieser API-Schlüssel hat keinen Zugriff auf die angefragte Ressource.",
            404:"Modell nicht gefunden oder für diesen API-Schlüssel nicht freigegeben. Modell-ID prüfen.",
            429:"OpenAI meldet ein Zugriffs- oder Anfragelimit. API-Konto und Limits prüfen."}.get(status,
            "OpenAI konnte die Anfrage nicht bestätigen. Bitte später erneut prüfen.")

def check_connection(api_key,model,provider=None):
    provider=resolve_provider(provider)
    if provider == "ollama":
        credentials = local_credentials(model.strip())
        with httpx.Client(timeout=3., trust_env=False) as client:
            return ollama.model_status(client, credentials.model)
    if provider == "claude":
        from .model_tests import function_test
        return function_test(api_key,model,provider,"text")
    credentials=resolve_draft(api_key,model,provider)
    if not credentials.model:raise ValueError("Bitte eine Modell-ID eintragen.")
    try:
        with httpx.Client(timeout=httpx.Timeout(12.,connect=5.)) as client:
            response=client.post("https://api.openai.com/v1/responses",
                headers={"Authorization":"Bearer "+credentials.api_key},
                json={"model":credentials.model,"input":"Hallo","max_output_tokens":64,"store":False})
        if response.status_code!=200:
            if response.status_code == 429:
                try:
                    error = response.json().get("error") or {}
                except (ValueError, AttributeError):
                    error = {}
                if isinstance(error, dict) and (error.get("code") == "insufficient_quota"
                        or "no credits remaining" in str(error.get("message", "")).lower()):
                    return {"ok":False,"status_code":429,"message":"OpenAI meldet fehlendes API-Guthaben oder ein ausgeschöpftes Budget."}
            return {"ok":False,"status_code":response.status_code,"message":_provider_error(response.status_code)}
        data = response.json()
        if isinstance(data, dict) and data.get("status") in ("completed", "incomplete"):
            return {"ok":True,"message":"OpenAI hat den Hallo-Test angenommen. API-Schlüssel und Modellzugriff funktionieren."}
        return {"ok":False,"message":"OpenAI hat keine gültige Testantwort geliefert."}
    except httpx.RequestError:
        return {"ok":False,"message":"OpenAI ist nicht erreichbar. Internetverbindung prüfen und erneut versuchen."}
    except ValueError:
        return {"ok":False,"message":"OpenAI hat keine lesbare Testantwort geliefert."}

def available_models(api_key,provider=None):
    provider=resolve_provider(provider)
    if provider == "ollama":
        return ollama.available_models()
    credentials=resolve_draft(api_key,"",provider)
    try:
        with httpx.Client(timeout=httpx.Timeout(12.,connect=5.)) as client:
            response=client.get(claude.BASE_URL+"/models" if provider=="claude" else "https://api.openai.com/v1/models",
                headers=claude.headers(credentials.api_key) if provider=="claude" else {"Authorization":"Bearer "+credentials.api_key})
        if response.status_code!=200:return {"ok":False,"message":f"{provider} meldet HTTP {response.status_code}. Zugang und Budget prüfen.","models":[]}
        try:items=response.json().get("data",[])
        except (ValueError,AttributeError):return {"ok":False,"message":"Modellliste konnte nicht gelesen werden.","models":[]}
        names=sorted({item.get("id") for item in items if isinstance(item,dict) and isinstance(item.get("id"),str)
                      and not item["id"].startswith("sk-") and re.fullmatch(r"[A-Za-z0-9._:/-]{1,200}",item["id"])})
        return {"ok":True,"models":names[:1000],"message":"Verfügbare Modell-IDs geladen. Ein Modell mit Bildinput und Function Calling auswählen."}
    except httpx.RequestError:
        return {"ok":False,"models":[],"message":f"{provider} ist nicht erreichbar. Internetverbindung prüfen."}

if __name__=="__main__":
    import sys
    if "--activate-provider" in sys.argv:
        try:
            index=sys.argv.index("--activate-provider")
            if index+1>=len(sys.argv):raise ValueError("Anbieter fehlt.")
            credentials=load_credentials(sys.argv[index+1])
            save_settings("",credentials.model,credentials.provider)
            print("Aktiver Anbieter: "+credentials.provider)
        except ValueError as error:
            print(str(error),file=sys.stderr)
            sys.exit(1)
    elif "--init" in sys.argv:print(initialize_settings())
    else:print(local_settings_path() if active_provider() == "ollama" else settings_path())
