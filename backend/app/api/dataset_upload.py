from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..models.synthetic_data_models import DatasetRow
from ..core.config import settings
import os
import pandas as pd
from datetime import datetime

router = APIRouter()

@router.post('/datasets/upload')
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Save file to /tmp for parsing
    temp_path = f"/tmp/{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    with open(temp_path, 'wb') as out_file:
        out_file.write(await file.read())
    try:
        df = pd.read_csv(temp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")
    os.remove(temp_path)

    # Store each row in DatasetRow table
    rows_added = 0
    for _, row in df.iterrows():
        db_row = DatasetRow(dataset_name=file.filename, row_data=row.to_dict())
        db.add(db_row)
        rows_added += 1
    db.commit()
    print(f"Uploaded dataset '{file.filename}' with {rows_added} rows.")

    return {"success": True, "rows_added": rows_added} 
