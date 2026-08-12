# routers/notifications.py
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from dependencies import get_db, get_current_user
from models import NotificationModel

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get("", response_model=List[dict])
def list_notifications(
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """List the current user's notifications, newest first."""
    notifs = (
        db.query(NotificationModel)
        .filter(NotificationModel.username == current_user)
        .order_by(NotificationModel.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": n.id,
            "type": n.type,
            "title": n.title,
            "message": n.message,
            "session_id": n.session_id,
            "is_read": n.is_read,
            "created_at": n.created_at.isoformat() if n.created_at else None,
        }
        for n in notifs
    ]


@router.get("/unread-count")
def get_unread_count(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Returns the total number of unread notifications for the user."""
    count = (
        db.query(NotificationModel)
        .filter(NotificationModel.username == current_user, NotificationModel.is_read == False)
        .count()
    )
    return {"unread_count": count}


@router.patch("/{notification_id}/read")
def mark_as_read(
    notification_id: int,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """Mark a specific notification as read."""
    notif = (
        db.query(NotificationModel)
        .filter(NotificationModel.id == notification_id, NotificationModel.username == current_user)
        .first()
    )
    if not notif:
        raise HTTPException(status_code=404, detail="Notification not found.")

    notif.is_read = True
    db.commit()
    return {"status": "success", "id": notification_id}