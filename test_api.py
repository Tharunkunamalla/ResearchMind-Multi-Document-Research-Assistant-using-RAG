import requests
import time

def test_api():
    base_url = "http://127.0.0.1:8000"
    
    # 1. Wait a moment for server to start
    print("Waiting for server to be responsive...")
    for _ in range(5):
        try:
            r = requests.get(f"{base_url}/health")
            if r.status_code == 200:
                print("Server is up and healthy!")
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    else:
        print("[ERROR] Server did not start in time.")
        return
        
    # 2. Test upload and parsing endpoint
    print("Sending POST request to parse PDF...")
    files = {'file': ('sample_paper.pdf', open('sample_paper.pdf', 'rb'), 'application/pdf')}
    r = requests.post(f"{base_url}/api/documents/parse", files=files)
    
    if r.status_code == 200:
        data = r.json()
        print("\n--- API Parse Success ---")
        print(f"Title: {data['title']}")
        print(f"Authors: {data['authors']}")
        print("\nSections parsed:")
        for section, content in data['sections'].items():
            print(f"- {section}: {len(content)} chars")
            
        # Assertions
        assert "A Study on Semantic Information Retrieval" in data["title"]
        assert len(data["authors"]) == 3
        assert "abstract" in data["sections"]
        print("\n[SUCCESS] API successfully parsed document and returned correct fields!")
    else:
        print(f"[FAIL] API returned status code {r.status_code}: {r.text}")

if __name__ == "__main__":
    test_api()
