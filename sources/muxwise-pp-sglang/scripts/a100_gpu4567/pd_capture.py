"""Strict normalized dual-role wire evidence, CPU only."""
import base64
import json
import re

def require(value, message):
    if not value: raise ValueError(message)

def validate_pd_capture_records(rows,pair,classifier):
    require(isinstance(rows,list) and len(rows)==16,'Every PD pair requires16 normalized captures')
    require(isinstance(pair,dict) and set(pair)=={'prefill','decode'}
            and all(isinstance(pair[role],dict) and pair[role].get('role')==role for role in pair),
            'Invalid saved PD pair roles')
    prefix=None
    for index,row in enumerate(rows):
        require(isinstance(row,dict),'Missing normalized PD capture')
        rid=row.get('requested_rid');room=row.get('bootstrap_room');responses=row.get('responses')
        require(isinstance(rid,str) and re.fullmatch(r'a100-cancel-\d+-\d+',rid),'Missing normalized cancellation RID')
        if prefix is None:prefix=rid.rsplit('-',1)[0]+'-'
        require(rid==prefix+str(index),'Missing or reordered cancellation RID')
        require(type(room) is int and 0<=room<2**63,'Missing shared bootstrap room')
        require(isinstance(responses,dict) and set(responses)=={'prefill','decode'},'Both PD role captures required')
        for role,response in responses.items():
            require(isinstance(response,dict) and 'error' not in response
                    and response.get('requested_rid')==rid and response.get('port')==pair[role]['port'],
                    'PD capture RID/port differs from saved pair')
            require(type(response.get('http_status')) is int and response['http_status'] in (200,500),
                    'Unexpected PD cancellation HTTP status')
            raw=response.get('raw_body');encoded=response.get('raw_body_base64')
            require(isinstance(raw,str) and isinstance(encoded,str),'Missing raw PD response bytes')
            require(base64.b64decode(encoded,validate=True).decode('utf-8',errors='replace')==raw,
                    'PD raw body/base64 mismatch')
            require(json.loads(raw)==response.get('response'),'PD raw body/parsed JSON mismatch')
            body=response['response'];require(isinstance(body,dict),'PD response body must be JSON object')
            if response['http_status']==500:
                rooms=re.findall(r'bootstrap_room=(\d+)',body.get('message',''))
                require(rooms and all(int(value)==room for value in rooms),'PD error body bootstrap room differs')
            else:
                meta=body.get('meta_info',{})
                require(isinstance(meta,dict) and meta.get('id')==rid,'PD normal body RID differs')
                # Normal bodies do not echo the room. The exact paired-call source
                # sends the same payload/outer room to both roles; validate any echo.
                require('bootstrap_room' not in meta or meta['bootstrap_room']==room,'PD normal body room differs')
        require(classifier(row,rid),'New exact role/stage cancellation predicate rejected saved capture')
