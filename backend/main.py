from fastapi import FastAPI

app = FastAPI(title="FunnelIQ")


@app.get("/")
def root():
    return {"message": "FunnelIQ backend is running"}


@app.get("/health")
def health():
    return {"status": "ok"}
