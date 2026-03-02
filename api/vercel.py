from index import app as vercel_app

# Vercel serverless function
def handler(request):
    return vercel_app(request)
