import React, {useState, useEffect} from 'react';
import '../styles/table.css';
import '../styles/player.css';
import Player from './Player.js'
import Button from 'react-bootstrap/Button';
import Card from './Card.js';
import { Container, Row, Col } from 'react-bootstrap';
import { Layout } from './Layout.js';
import Cookies from 'js-cookie';

function loadDetails(callback, url){
    const xhr = new XMLHttpRequest()
    const method = 'GET'
    const responseType = "json"
    xhr.responseType = responseType
    xhr.open(method, url)
    xhr.setRequestHeader('Authorization', 'Token caf80ff750f4fce1d5f58b6f79a90cc0ef47614c')
    xhr.onload = function(){
        callback(xhr.response, xhr.status)
    }
    xhr.onerror = function(){
        callback({
            "message": "The request was an error"
        }, 400)
    }
    xhr.send()
}


function Table(props){
    const [pot, setPot] = useState("")
    const [loaded, setLoaded] = useState(false)
    const [communityCards, setCommunityCards] = useState([
        {id: 0, card_str: '9c', game: 0},
    ])
    const [players, setPlayers] = useState([
        {
            "id": "",
            "name": "",
            "stack": "",
            "games": [],
            "cards": []
        }
    ])

    useEffect(() => {
        const gamesCallback = (response, status) => {
            if(status === 200){
                const data = response.slice(response.length-1)[0]
                setPot(data["total_pot"])
            }
        }
        loadDetails(gamesCallback, "http://localhost:8000/api/games/")

        const playersCallback = (response, status) => {
            if(status === 200){
                const playerGuest = response.slice(response.length-2)[0]
                const playerAI = response.slice(response.length-1)[0]
                console.log("PLAYERS LOG")
                console.log({playerGuest, playerAI})
                setPlayers({playerGuest, playerAI})
                
            }
        }
        loadDetails(playersCallback, "http://localhost:8000/api/players/")

        const communityCardsCallback = (response, status) => {
            if(status === 200){
                // console.log(response)
                setCommunityCards(response)
                // console.log(communityCards)
            }
        }
        loadDetails(communityCardsCallback, `http://localhost:8000/api/community_cards/`)

    }, [])

    return (
        <div>
            <Layout>
                <Row>
                    <Col>
                        <div className="heading">
                            <h1>Texas Hold'Em</h1>
                        </div>
                    </Col>
                </Row>
                <div className="table">
                    <div id="board" className="board">
                        {communityCards.map((card, index) => (
                            <Card 
                                key={index}
                                index={index}
                                value={card.id}
                            />
                        ))}
                    </div>
                    <Row>
                        <Col>
                            <h1 style={{textAlign: "center"}}>Pot: ${pot}</h1>
                        </Col>
                    </Row>
                    
                    <Row>   
                        {players.map((player, index) => (
                            <Card 
                                key={index}
                                index={index}    
                                value={player.card_str}
                                suit={player.card_str}
                            />
                        ))}

                        <Col>   
                            <Player greeting={"Welcome to React"} is_ai={true} />
                        </Col>  
                    </Row>  

                    <Row>
                        <div className="mx-auto">
                            <Button variant="success" size="lg" block>Check/Call</Button>
                            <Button variant="danger" size="lg" block>Bet/Raise</Button>
                            <Button variant="light" size="lg" block>Fold</Button>
                        </div>
                    </Row>
                </div>
            </Layout>
        </div>

    )

}

export default Table;