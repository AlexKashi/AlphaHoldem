import React from 'react'
import '../styles/table.css';

function format_card(suit){
    if(suit === 'c'){
        return ['♣', 'black'];
    }else if(suit === 's'){
        return ['♠', 'black'];
    }else if(suit === 'd'){
        return ['♦', 'red'];
    }else if(suit === 'h'){
        return ['♥', 'red'];
    }else{
        return []
    }
}
function Card(props) {
    // console.log(props)
    const [suit, colour] = format_card(props.suit)
    const suit_element = <p class={`card-img ${colour}`}>{suit}</p>;
    return (
        <div class="card-small">
            <p class={`card-text ${colour}`}>{props.value}</p>
            {suit_element}
        </div>
    )
}

export default Card