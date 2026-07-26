from fastapi import FastAPI

app = FastAPI(title="FunnelIQ")


@app.get("/health")
def health():
    return {"status": "ok"}
