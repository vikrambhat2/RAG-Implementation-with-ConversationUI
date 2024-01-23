css = '''
<style>
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    display: flex;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    transition: 0.3s;
}

.chat-message.user {
    background-color: #4a69bd;
    color: white;
    align-items: center;
}

.chat-message.bot {
    background-color: #78e08f;
    color: black;
    align-items: center;
}

.chat-message .avatar {
    width: 15%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.chat-message .avatar img {
    max-width: 50px;
    max-height: 50px;
    border-radius: 50%;
    object-fit: cover;
}

.chat-message .message {
    width: 85%;
    padding: 0 1rem;
    font-size: 1rem;
    font-family: 'Arial', sans-serif;
}

.chat-message:hover {
    box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
}
</style>
'''

bot_template = '''
<div class="chat-message bot"">
    <div class="avatar">
        <img src="https://uxwing.com/wp-content/themes/uxwing/download/communication-chat-call/answer-icon.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user" style="text-align: left;">
    <div class="avatar">
        <img src="https://cdn.iconscout.com/icon/free/png-512/free-q-characters-character-alphabet-letter-36051.png?f=webp&w=512">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
