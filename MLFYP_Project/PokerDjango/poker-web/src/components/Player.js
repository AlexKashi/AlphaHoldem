import React, {useState, useEffect} from 'react';
import '../styles/player.css';
import Card from './Card.js'


function Player(props) {
    
    let which_player = props.is_ai? "player2": "player1";
    let tag = which_player==="player2"? "AI": "Guest";
    const message = props => <h1>{props.greeting}</h1>;
    useEffect(() => {
        // const playerCardsCallback = (response, status) => {
        //     if(status === 200){
        //         console.log("playerGuestCardsCallback", response)
        //         setPlayer({...player, response})
        //     }
        // }
        // console.log("PLAYER cards URL", `http://localhost:8000/api/players/${player.id}/display_player_cards/`)
        // // loadDetails(playerGuestCardsCallback, `http://localhost:8000/api/players/${player.id}/display_player_cards/`)

    }, [])
    return (
        <div className={which_player}>
            <div className="holding">
                {message}
                <Card value={'A'} suit={'d'}/>
                <Card value={'K'} suit={'s'}/>
            </div>
            

            <p className="playerTag">{message}</p>
        </div>
    )
}

export default Player;