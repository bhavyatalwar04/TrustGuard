# backend/app/services/alert_system.py

import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging
from sqlalchemy.orm import Session
from ..database import get_db
from ..models.claim import Claim
from ..models.verification import Verification
from ..config import settings

logger = logging.getLogger(__name__)

class AlertSystem:
    def __init__(self):
        self.email_config = {
            'smtp_server': settings.SMTP_SERVER,
            'smtp_port': settings.SMTP_PORT,
            'username': settings.SMTP_USERNAME,
            'password': settings.SMTP_PASSWORD,
            'from_email': settings.FROM_EMAIL
        }
        self.alert_thresholds = {
            'high_misinformation': 0.8,
            'trending_false_claims': 5,
            'rapid_spread': 100
        }

    async def monitor_claims(self):
        """Continuously monitor claims for alert conditions"""
        while True:
            try:
                db = next(get_db())
                await self._check_misinformation_levels(db)
                await self._check_trending_false_claims(db)
                await self._check_rapid_spread(db)
                db.close()
                
                # Check every 5 minutes
                await asyncio.sleep(300)
            except Exception as e:
                logger.error(f"Error in claim monitoring: {str(e)}")
                await asyncio.sleep(60)

    async def _check_misinformation_levels(self, db: Session):
        """Check for high levels of misinformation"""
        recent_time = datetime.utcnow() - timedelta(hours=1)
        
        recent_verifications = db.query(Verification).filter(
            Verification.created_at >= recent_time
        ).all()
        
        if len(recent_verifications) < 10:
            return
        
        false_count = sum(1 for v in recent_verifications if v.verdict == 'FALSE')
        misinformation_ratio = false_count / len(recent_verifications)
        
        if misinformation_ratio >= self.alert_thresholds['high_misinformation']:
            await self._send_alert(
                'High Misinformation Alert',
                f'Detected {misinformation_ratio:.2%} misinformation in the last hour',
                'high_misinformation',
                {'ratio': misinformation_ratio, 'total_claims': len(recent_verifications)}
            )

    async def _check_trending_false_claims(self, db: Session):
        """Check for trending false claims"""
        recent_time = datetime.utcnow() - timedelta(hours=6)
        
        false_claims = db.query(Claim).join(Verification).filter(
            Claim.created_at >= recent_time,
            Verification.verdict == 'FALSE'
        ).all()
        
        # Group by similar content (simplified)
        claim_groups = {}
        for claim in false_claims:
            key = claim.text[:50]  # Simple grouping by first 50 chars
            if key not in claim_groups:
                claim_groups[key] = []
            claim_groups[key].append(claim)
        
        for group_key, claims in claim_groups.items():
            if len(claims) >= self.alert_thresholds['trending_false_claims']:
                await self._send_alert(
                    'Trending False Claim Alert',
                    f'False claim spreading rapidly: "{group_key}..."',
                    'trending_false',
                    {'claim_count': len(claims), 'sample_text': group_key}
                )

    async def _check_rapid_spread(self, db: Session):
        """Check for rapidly spreading claims"""
        recent_time = datetime.utcnow() - timedelta(minutes=30)
        
        recent_claims = db.query(Claim).filter(
            Claim.created_at >= recent_time
        ).all()
        
        for claim in recent_claims:
            if claim.engagement_score and claim.engagement_score >= self.alert_thresholds['rapid_spread']:
                await self._send_alert(
                    'Rapid Spread Alert',
                    f'Claim spreading rapidly with engagement score: {claim.engagement_score}',
                    'rapid_spread',
                    {'claim_id': claim.id, 'engagement_score': claim.engagement_score}
                )

    async def _send_alert(self, subject: str, message: str, alert_type: str, metadata: Dict[str, Any]):
        """Send alert via email and log"""
        logger.warning(f"ALERT [{alert_type}]: {subject} - {message}")
        
        # Store alert in database
        db = next(get_db())
        # Add alert model if needed
        
        # Send email notification
        await self._send_email_alert(subject, message, metadata)

    async def _send_email_alert(self, subject: str, message: str, metadata: Dict[str, Any]):
        """Send email alert to administrators"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = settings.ADMIN_EMAIL
            msg['Subject'] = f"TruthGuard Alert: {subject}"
            
            body = f"""
            Alert: {subject}
            
            Message: {message}
            
            Timestamp: {datetime.utcnow().isoformat()}
            
            Metadata: {metadata}
            
            This is an automated alert from TruthGuard system.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            text = msg.as_string()
            server.sendmail(self.email_config['from_email'], settings.ADMIN_EMAIL, text)
            server.quit()
            
            logger.info(f"Alert email sent successfully: {subject}")
        except Exception as e:
            logger.error(f"Failed to send alert email: {str(e)}")

    async def send_custom_alert(self, recipients: List[str], subject: str, message: str):
        """Send custom alert to specific recipients"""
        for recipient in recipients:
            try:
                msg = MIMEMultipart()
                msg['From'] = self.email_config['from_email']
                msg['To'] = recipient
                msg['Subject'] = subject
                
                msg.attach(MIMEText(message, 'plain'))
                
                server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
                server.starttls()
                server.login(self.email_config['username'], self.email_config['password'])
                text = msg.as_string()
                server.sendmail(self.email_config['from_email'], recipient, text)
                server.quit()
                
                logger.info(f"Custom alert sent to {recipient}")
            except Exception as e:
                logger.error(f"Failed to send custom alert to {recipient}: {str(e)}")

# Global alert system instance
alert_system = AlertSystem()