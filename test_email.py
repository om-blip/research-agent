import asyncio
from mcp.tools.email_tool import send_email

async def main():
    result = await send_email(
        to="ommuddebihal@gmail.com",
        subject="Research Agent Test Email",
        body_markdown="""
# Test Report

This is a test email from your Research Agent.

## It works if you see this

- RAG pipeline: working
- Groq synthesis: working  
- Email delivery: working

**All systems operational.**
        """
    )
    print(f"Result: {result}")

asyncio.run(main())