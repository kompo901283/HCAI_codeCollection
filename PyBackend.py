import torch
import json
import os
import random
import stanza
import Levenshtein
import logging
import re
from deep_translator import GoogleTranslator
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Logging to see whats happening
logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELpath = "TestSave5"
TRANSCACHEfile = "translation_cache.json"
UICACHEfile = "ui_translations_cache.json"
SCORESdir = "user_data"

# Balance settings for the bucket system
MAXscore = 5.0
MINscore = -5.0
PENALTYfactor = 0.3 

UIdefaults = { # Low-tech but safe method of storing UI defaults
    "title": "B.I.S.O.N", 
    "profile": "Mastery Profile", 
    "start": "Run", 
    "submit": "Submit Answer",
    "next": "Next Challenge ->",
    "help": "Help (Hint)", 
    "peek": "Peek Translation", 
    "manageData": "Manage My Data",
    "howGraded": "How am I graded?", 
    "gradingExplanation": "We compare your answer to the target using Levenshtein distance (edit distance). 0 means perfect. 1-2 means a typo. Higher means a different word. Your mastery score updates based on this distance.", 
    "confirmDelete": "Are you sure you want to delete all data? This cannot be undone.", 
    "youWrote": "You wrote:",
    "correct": "Correct:",
    "correctExcl": "Correct!",
    "incorrectExcl": "Incorrect",
    "thinking": "Thinking...",
    "connectionFail": "Connection failed. Is the server running?",
    "noData": "No data yet. Press Run!",
    "backendError": "Backend Error: ",    
    "infoTitle": "How B.I.S.O.N Works",
    "infoT5Header": "The T5 Model",
    "infoT5Text": "We use a specialized AI model (Google-MT5-base) which has been trained specifically to generate sentences in Norwegian! It has essentially been shown many examples of words and their labels, which means that we can show it what labels you want to practice, and it will create new sentences which contain those words for you to practice on.",
    "infoClustersHeader": "Magic Clusters",
    "infoClustersText": "The labels are divided into 2 parts. One type of label focuses only on grammatical rules (Eg; \"NOUN\", \"obj\"), whilst the other part is the \"magic\" labels. These are called magic labels beacuse there is no way of truly knowing why they are associated with each other. This is because the patterns are those which are found only in the \"brain\" of the AI, similarily to how humans also have their individual ways of thinkng. Additionally, the AI has incredibly good pattern discovery skills, and since its a computer program, we are capable of actually storing these patterns. This means that when you see 2 words which look completely different, but are associated with the same magic label, then there is a hidden correlation between those words, which was found by the AI.",
    "infoDataHeader": "Your Data",
    "infoDataText": "The only data stored by our program is your scores for each individual label and your username / password. This is done to store your progress, and to ensure that the AI can create questions which are relevant to you. All your mastery scores are displayed on the screen, but can also be viewed or deleted by using the \"Manage My Data\" button.",
    "infoGotIt": "Got it!",    
    "tutorialBtn": "Tutorial",
    "tutorialTitle": "How to use B.I.S.O.N",
    "tutIntro": "In order to start, press the \"run\" button. Then you will be faced by a sentence with one or more missing words. Your task is to fill in these words. The missing word in the sentence is associated with one or more labels. Labels are essentially groups of words which have some sort of similarity.",
    "tutProgress": "As you progress, your scores for individual labels will be placed into 5 different tiers of mastery based on your success with words associated with those labels. You can then choose which of these tiers you wish to practice.",
    "tutRec": "It is recommended to use the exploration method (tier 3) for at least 15 tasks. That way you will have been graded on enough groups of words that you can start practicing those you scored the worst on, or maybe rehearse those you previously did good on.",
    "tutGroups": "There are nearly 70 different groups of words when you include those which are based purely on grammar rules. That is a lot groups to explore!",
    "tutGoodToKnowHeader": "Good to know:",
    "tutGoodToKnow1": "You can delete your data and your account through the \"Manage My Data\" button.",
    "tutGoodToKnow2": "You can view other words related to labels which you are curious about by clicking the \"View Cluster Dictionary\" button.",
    "tutGoodToKnow3": "You can change language, difficulty or ask for hints by using the buttons surrounding the text field.",
    "viewMap": "View Cluster Dictionary",
    "dataModalTitle": "My Data Files",
    "dataModalText": "Currently, the data related to your mastery of labels is the only data that is stored and tied to your account, apart from your username and password.",
    "deleteLocal": "Delete local data and user",
    "deleteFile": "Delete This File"
}

class InferenceEngine:
    def __init__(self):
        if not os.path.exists(SCORESdir): os.makedirs(SCORESdir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODELpath)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(MODELpath).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"model load failed... {e}")
            raise e

        # Stanza is heavy but needed for checks
        logger.info("Loading Stanza Pipeline...")
        self.nlp = stanza.Pipeline("no", processors = "tokenize,pos", verbose=False)
        
        # Load up the grammar mapping files
        self.clusterMap = self._loadClusterMap()
        self.validLabels = self._generateLabels()
        
        self.transCache = self._loadJson(TRANSCACHEfile)
        self.uiCache = self._loadJson(UICACHEfile)
        
        if "en" not in self.uiCache:
            self.uiCache["en"] = UIdefaults
            self._saveJson(UICACHEfile, self.uiCache)

        self.sessions = {} # RAM storage for active games
        logger.info(f"System ready, loaded {len(self.validLabels)} valid labels...")

    def _loadJson(self, path):
        return json.load(open(path, "r", encoding="utf-8")) if os.path.exists(path) else {}

    def _saveJson(self, path, data):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent= 2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed saving JSON {path}: {e}")

    def _getUserScoresPath(self, username):
        safeName = "".join([c for c in username if c.isalnum() or c in ('-', '_')])
        if not safeName: safeName = "default"
        return os.path.join(SCORESdir, f"scores_{safeName}.json")

    def getUserScores(self, username):
        return self._loadJson(self._getUserScoresPath(username))

    def saveUserScores(self, username, scores):
        self._saveJson(self._getUserScoresPath(username), scores)

    # Deletes the users score file
    def deleteUserData(self, username):
        path = self._getUserScoresPath(username)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def listUserFiles(self, username):
        safeName = "".join([c for c in username if c.isalnum() or c in ('-', '_')])
        target = f"scores_{safeName}.json"
        if os.path.exists(os.path.join(SCORESdir, target)):
            return [{"filename": target, "type": "Score Data", "path": os.path.join(SCORESdir, target)}]
        return []

    # Reads the content of a specific user file
    def getFileContent(self, filename):
        if "/" in filename or "\\" in filename: return None
        path = os.path.join(SCORESdir, filename)
        if os.path.exists(path):
            with open(path, 'r', encoding = 'utf-8') as f:
                return f.read()
        return None

    def _loadClusterMap(self):
        path = os.path.join(MODELpath, "cluster_map_cleaned.json")
        if os.path.exists(path):
            return json.load(open(path, "r", encoding = "utf-8"))
        logger.warning("No cluster map found...")
        return {}

    def getSortedClusterMap(self):
        if not self.clusterMap: return "No cluster map loaded."
        
        groups = {}
        for word, cid in self.clusterMap.items():
            if cid not in groups: groups[cid] = []
            groups[cid].append(word)
        
        # Sort by id
        sorted_ids = sorted(groups.keys())
        
        lines = []
        for cid in sorted_ids:
            words = sorted(groups[cid])
            lines.append(f"-> Cluster {cid} ({len(words)} words) <-" )
            lines.append(", ".join(words))
            lines.append("")
            
        return "\n".join(lines)

    def _generateLabels(self):
        labels = []
        if self.clusterMap:
            groups = {}
            for w, cid in self.clusterMap.items():
                groups.setdefault(cid, []).append(w)
            for cid, words in groups.items():
                if len(words) > 0: labels.append(f"magic_cluster {cid}")
        for tag in ["NOUN", "VERB", "ADJ", "ADV"]:
            labels.append(f"upos {tag}")
        return labels

    # Google translate
    def translateText(self, text, targetLang='en'):
        cacheKey = f"{text}_{targetLang}"
        if cacheKey in self.transCache: return self.transCache[cacheKey]
        try:
            trans = GoogleTranslator(source='no', target=targetLang).translate(text)
            self.transCache[cacheKey] = trans
            self._saveJson(TRANSCACHEfile, self.transCache)
            return trans
        except Exception as e:
            logger.error(f"Translation API failed: {e}")
            return f"({text})"

    def getUiStrings(self, lang):
        if lang in self.uiCache:
            cached = self.uiCache[lang]
            missingKeys = [k for k in UIdefaults if k not in cached]
            
            # if we have partial data, fill in the blanks
            if not missingKeys:
                return cached
            
            logger.info(f"Updating UI cache for '{lang}' with {len(missingKeys)} new keys...")
            try:
                translator = GoogleTranslator(source='en', target=lang)
                missingValues = [UIdefaults[k] for k in missingKeys]
                translatedMissing = translator.translate_batch(missingValues)
                
                for k, v in zip(missingKeys, translatedMissing):
                    cached[k] = v
                
                self.uiCache[lang] = cached
                self._saveJson(UICACHEfile, self.uiCache)
                return cached
            except Exception as e:
                logger.error(f"Partial UI update failed for {lang}... {e}")
                for k in missingKeys:
                    cached[k] = UIdefaults[k]
                return cached

        try:
            translator = GoogleTranslator(source= "en", target = lang) # Currently lang can only be 13 languages, but the library supports many more
            keys = list(UIdefaults.keys())
            values = list(UIdefaults.values())
            translatedValues = translator.translate_batch(values)
            newUi = dict(zip(keys, translatedValues))
            self.uiCache[lang] = newUi
            self._saveJson(UICACHEfile, self.uiCache)
            return newUi
        except Exception as e:
            return UIdefaults

    def selectLabelsByDifficulty(self, difficulty, userScores):
        def filterFn(lbl, score):
            if difficulty == "1": return score >= 3.5 # Mastered
            if difficulty == "2": return 1.5 <= score < 3.5
            if difficulty == "3": return -1.5 <= score < 1.5 # Neutral
            if difficulty == "4": return -3.5 <= score < -1.5
            if difficulty == "5": return score < -3.5 # Hard
            return True

        candidates = [l for l in self.validLabels if filterFn(l, userScores.get(l, 0))]
        
        # If the bucket is empty, use mid ladder bucket in order to map out struggles
        if not candidates and difficulty != "3":
            logger.info(f"Bucket {difficulty} empty, falling back to exploration...")
            candidates = [l for l in self.validLabels if -1.5 <= userScores.get(l, 0) < 1.5]
            
        if not candidates: 
            candidates = self.validLabels
            
        random.shuffle(candidates)
        return candidates

    # The core logic, pick a topic, generate a sentence, validate it
    def generateChallenge(self, difficulty, targetLang, username):
        userScores = self.getUserScores(username)
        candidates = self.selectLabelsByDifficulty(difficulty, userScores)
        logger.info(f"Bucket generation, diff: {difficulty}, User: {username}")

        maxGenRetries = 3
        
        for labelIdx in range(min(len(candidates), 3)):
            targetLabel = candidates[labelIdx]
            logger.info(f"Trying label '{targetLabel}'...") 
            
            promptText = targetLabel
            if "magic_cluster" in targetLabel:
                try:
                    cID = int(targetLabel.split()[1])
                    cWords = [w for w, i in self.clusterMap.items() if i == cID]
                    if cWords:
                        forcedWord = random.choice(cWords)
                        promptText = f"construct sentence with: {forcedWord}"
                except:
                    pass

            inputIds = self.tokenizer(promptText, return_tensors="pt").input_ids.to(self.device)

            # try generating a valid sentence for the label
            for attempt in range(maxGenRetries):
                out = self.model.generate(
                    inputIds, max_length = 64, do_sample = True, top_k = 60, temperature = 0.85 
                )
                text = self.tokenizer.decode(out[0], skip_special_tokens=True)
                if len(text) < 10: 
                    logger.debug("Text too short, skipping...") 
                    continue
                
                validMatch = self._validateAndExtract(text, targetLabel)
                if validMatch:
                    logger.info(f"Succesfully generated valid match on attempt {attempt+1}...") 
                    return self._finalizeSession(text, validMatch, targetLabel, difficulty, targetLang, username)
                else:
                    logger.debug(f"Failed validation for {text}") 

        logger.warning("Complex generation failed, falling back to easy labels...")
        return self._forceFallbackGeneration(targetLang, username)

    # Uses Stanza to check which word matches the target label
    def _validateAndExtract(self, text, label):
        try:
            doc = self.nlp(text)
            reqCluster = int(label.split()[1]) if "magic_cluster" in label else None
            reqPos = label.split()[1] if "upos" in label else None

            matches = []
            for sent in doc.sentences:
                for w in sent.words:
                    isMatch = False
                    if reqCluster is not None:
                        if self.clusterMap.get(w.text) == reqCluster: isMatch = True
                    if reqPos and w.upos == reqPos: isMatch = True
                    if isMatch: matches.append(w.text)
            return random.choice(matches) if matches else None
        except Exception as e:
            logger.error(f"Validation error, {e}")
            return None

    # Since the model can fail to generate sentences that we accept with cluster labels, we fall back to basic grammar labels if that happens
    def _forceFallbackGeneration(self, targetLang, username):
        try:
            safeLabels = ["upos NOUN", "upos VERB", "upos ADJ", "upos ADV"]
            lbl = random.choice(safeLabels)
            reqPos = lbl.split()[1]
            
            inputIds = self.tokenizer(lbl, return_tensors="pt").input_ids.to(self.device)
            out = self.model.generate(inputIds, max_length = 64, do_sample = True)
            text = self.tokenizer.decode(out[0], skip_special_tokens = True)
            
            doc = self.nlp(text)
            matches = [w.text for s in doc.sentences for w in s.words if w.upos == reqPos]
            target = random.choice(matches) if matches else text.split()[-1] 
            
            return self._finalizeSession(text, target, lbl, "FALLBACK", targetLang, username)
        except Exception as e:
            return {"error": f"Fallback failed: {str(e)}"}

    # prepare the payload for the frontend
    def _finalizeSession(self, text, targetWord, label, diff, lang, username):
        wordTrans = self.translateText(targetWord, lang)
        sentTrans = self.translateText(text, lang)
        
        pattern = re.compile(rf"\b{re.escape(wordTrans)}\b", re.IGNORECASE)
        if pattern.search(sentTrans):
            finalHint = pattern.sub(f"<b>{wordTrans}</b>", sentTrans)
        else:
            finalHint = f"{sentTrans} (Target: <b>{wordTrans}</b>)"

        # Mask logic
        maskedHintTrans = pattern.sub("[___]", sentTrans)
        if maskedHintTrans == sentTrans: 
             maskedHintTrans = f"{sentTrans} (Target: [___])"

        sessId = str(random.randint(10000, 99999))
        patternMask = re.compile(rf"\b{re.escape(targetWord)}\b", re.IGNORECASE)
        maskedText = patternMask.sub("[___]", text)
        
        self.sessions[sessId] = {
            "target_word": targetWord,
            "target_label": label,
            "hint": finalHint,
            "full_text": text,
            "username": username
        }
        return {
            "session_id": sessId,
            "text": maskedText,
            "translated_word": wordTrans,
            "translated_sentence": finalHint, 
            "masked_translated_sentence": maskedHintTrans, 
            "label_type": label,
            "difficulty_served": diff,
            "target_word_clean": targetWord
        }

    # We grade the users input with Levenshtein, we should implement a better method
    def gradeAnswer(self, sessionId, userAnswer, username):
        session = self.sessions.get(sessionId)
        if not session: return {"error": "Session expired"}
        
        if session.get("username") != username: return {"error": "User mismatch"}

        target = session["target_word"]
        label = session["target_label"]
        dist = Levenshtein.distance(userAnswer.lower().strip(), target.lower().strip())
        
        status = "incorrect"
        scoreChange = 0.0
        
        if dist == 0:
            status = "correct"
            scoreChange = 1.0 
        else:
            punishment = dist * PENALTYfactor
            scoreChange = -punishment
            if dist <= 2: status = "close"
        
        # Update user stats
        userScores = self.getUserScores(username)
        oldScore = userScores.get(label, 0)
        newScore = max(MINscore, min(MAXscore, oldScore + scoreChange))
        userScores[label] = round(newScore, 2)
        self.saveUserScores(username, userScores)
        
        translatedSentence = session["hint"]
        
        # Clean up session
        del self.sessions[sessionId]
        
        return {
            "status": status,
            "correct_word": target,
            "user_word": userAnswer,
            "distance": dist,
            "score_change": round(scoreChange, 2),
            "new_label_score": userScores[label],
            "scores_snapshot": userScores,
            "label": label,
            "translated_sentence": translatedSentence,
            "full_sentence": session["full_text"]
        }

app = Flask(__name__)
CORS(app)
engine = InferenceEngine()


@app.route("/scores", methods = ["POST"])
def endpointScores():
    username = request.json.get("username", "default")
    return jsonify(engine.getUserScores(username))

@app.route("/generate", methods = ["POST"])
def endpointGenerate():
    data = request.json
    return jsonify(engine.generateChallenge(
        data.get("difficulty", "3"), 
        data.get("lang", "en"),
        data.get("username", "default")
    ))

@app.route("/grade", methods = ["POST"])
def endpointGrade():
    data = request.json
    return jsonify(engine.gradeAnswer(
        data.get("session_id"),
        data.get("answer", ""),
        data.get("username", "default")
    ))

@app.route("/localize", methods = ["POST"])
def endpointLocalize():
    lang = request.json.get("lang", "en")
    return jsonify(engine.getUiStrings(lang))

@app.route("/list_user_data", methods = ["POST"])
def endpointListUserData():
    username = request.json.get("username")
    if not username: return jsonify({"error": "Missing username"}), 400
    return jsonify(engine.listUserFiles(username))

@app.route("/get_file_content", methods = ["POST"])
def endpointGetFileContent():
    filename = request.json.get("filename")
    if not filename: return jsonify({"error": "Missing filename"}), 400
    content = engine.getFileContent(filename)
    if content is None: return jsonify({"error": "File not found"}), 404
    return jsonify({"content": content})

@app.route("/get_cluster_map", methods = ["GET"])
def endpointGetClusterMap():
    return jsonify({"content": engine.getSortedClusterMap()})

@app.route("/delete_user_data", methods = ["POST"])
def endpointDeleteUserData():
    username = request.json.get("username")
    if not username: return jsonify({"error": "Missing username"}), 400
    
    success = engine.deleteUserData(username)
    if success:
        return jsonify({"status": "deleted", "message": "User score data deleted."})
    else:
        return jsonify({"status": "not_found", "message": "No data found for this user."})

if __name__ == "__main__":
    app.run(port= 5000)
