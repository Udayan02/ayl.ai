import yt_dlp

# URL of the YouTube video
video_url = "https://youtube.com/clip/Ugkx1bs6Q7y1Pt1_y2N7L7fPCxjaI55cT7KJ?feature=shared"

# Define options for yt-dlp
ydl_opts = {
    'format': 'bestvideo+bestaudio/best',  # Downloads best video and audio quality
    'outtmpl': '%(title)s.%(ext)s',  # Output filename format
}

# Download video
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([video_url])
